const DEFAULT_SETTINGS = {
    apiBaseUrl: "https://avijit070-spam-email-detection.hf.space",
    apiKey: "",
    requestTimeoutMs: 60000,
    autoScanEnabled: true,
    historyLimit: 12
};

const predictionCache = new Map();
const pendingAnalyses = new Map();
const CACHE_TTL_MS = 90 * 1000;
const CACHE_MAX_SIZE = 50;
let historyLock = Promise.resolve();

chrome.runtime.onInstalled.addListener((details) => {
    const init = async () => {
        const settings = await getSettings();
        let changed = false;
        const needsMigration = settings.apiBaseUrl === "http://127.0.0.1:8000"
            || settings.apiBaseUrl === "http://localhost:8000"
            || settings.apiBaseUrl === "http://localhost"
            || settings.apiBaseUrl === "http://127.0.0.1";
        if (needsMigration) {
            settings.apiBaseUrl = DEFAULT_SETTINGS.apiBaseUrl;
            changed = true;
        }
        if (details.reason === "install") {
            changed = true;
        }
        if (changed) {
            await chrome.storage.sync.set({ settings });
        }
    };
    init().catch((error) => {
        console.error("Failed to initialize settings:", error);
    });
});

function normalizePayload(payload) {
    const safe = (payload && typeof payload === "object") ? payload : {};
    return {
        sender: typeof safe.sender === "string" ? safe.sender.trim() : "",
        subject: typeof safe.subject === "string" ? safe.subject.trim() : "",
        body: typeof safe.body === "string" ? safe.body.trim() : ""
    };
}

function cacheKey(payload) {
    return JSON.stringify([
        (payload.sender || "").toLowerCase(),
        (payload.subject || "").toLowerCase(),
        (payload.body || "").toLowerCase()
    ]);
}

function evictCacheIfNeeded() {
    const now = Date.now();
    for (const [key, cached] of predictionCache) {
        if (now - cached.timestamp > CACHE_TTL_MS) {
            predictionCache.delete(key);
        }
    }
    if (predictionCache.size > CACHE_MAX_SIZE) {
        const oldestKeys = [...predictionCache.keys()].slice(0, predictionCache.size - CACHE_MAX_SIZE);
        oldestKeys.forEach((key) => predictionCache.delete(key));
    }
}

function getCachedPrediction(key) {
    const cached = predictionCache.get(key);
    if (!cached) {
        return null;
    }

    if ((Date.now() - cached.timestamp) > CACHE_TTL_MS) {
        predictionCache.delete(key);
        return null;
    }

    return cached.value;
}

async function getSettings() {
    try {
        const data = await chrome.storage.sync.get("settings");
        const stored = data.settings || {};
        return {
            ...DEFAULT_SETTINGS,
            ...stored
        };
    } catch (error) {
        console.error("Failed to load settings:", error);
        return { ...DEFAULT_SETTINGS };
    }
}

function normalizeApiBaseUrl(url) {
    const value = String(url || "").trim().replace(/\/+$/, "");
    const normalized = value || DEFAULT_SETTINGS.apiBaseUrl;
    try {
        const parsed = new URL(normalized);
        if (parsed.protocol === "https:") {
            return parsed.origin;
        }
        if (parsed.protocol === "http:" && ["localhost", "127.0.0.1"].includes(parsed.hostname)) {
            return parsed.origin;
        }
        throw new Error("Use http:// only for localhost, or https:// for deployed backends.");
    } catch (error) {
        if (error.message.includes("Use http")) {
            throw error;
        }
        throw new Error("Please enter a valid URL (e.g., https://example.com)");
    }
}

async function saveSettings(partialSettings = {}) {
    const current = await getSettings();
    const merged = {
        ...current,
        ...partialSettings
    };

    merged.apiBaseUrl = normalizeApiBaseUrl(merged.apiBaseUrl);
    merged.apiKey = typeof merged.apiKey === "string" ? merged.apiKey.trim() : "";
    merged.requestTimeoutMs = Math.max(2000, Math.min(60000, Number(merged.requestTimeoutMs) || DEFAULT_SETTINGS.requestTimeoutMs));
    merged.historyLimit = Math.max(5, Math.min(50, Number(merged.historyLimit) || DEFAULT_SETTINGS.historyLimit));
    merged.autoScanEnabled = Boolean(merged.autoScanEnabled);

    await chrome.storage.sync.set({ settings: merged });
    return merged;
}

async function fetchJson(path, options = {}, settingsOverride = null) {
    const settings = settingsOverride || await getSettings();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), settings.requestTimeoutMs);

    const headers = { ...(options.headers || {}) };
    if (settings.apiKey) {
        headers["X-API-Key"] = settings.apiKey;
    }

    try {
        const response = await fetch(`${settings.apiBaseUrl}${path}`, {
            ...options,
            headers,
            signal: controller.signal
        });

        const contentType = response.headers.get("content-type") || "";
        let body;
        try {
            body = contentType.includes("application/json")
                ? await response.json()
                : await response.text();
        } catch (parseError) {
            if (parseError instanceof SyntaxError) {
                throw new Error("Backend returned invalid JSON.");
            }
            throw parseError;
        }

        if (!response.ok) {
            const detail = typeof body === "object" && body && "detail" in body
                ? body.detail
                : body || `Request failed with status ${response.status}`;
            throw new Error(String(detail));
        }

        return body;
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Backend request timed out. Check that the FastAPI server is running.");
        }
        if (error instanceof TypeError) {
            throw new Error(`Could not connect to backend at ${settings.apiBaseUrl}. Check your network and server status.`);
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

async function getScanHistory() {
    const data = await chrome.storage.local.get("scanHistory");
    return Array.isArray(data.scanHistory) ? data.scanHistory : [];
}

async function saveScanHistory(history) {
    await chrome.storage.local.set({ scanHistory: history });
}

async function pushHistoryEntry(payload, prediction) {
    await withHistoryLock(async () => {
        const settings = await getSettings();
        const history = await getScanHistory();
        const nextEntry = {
            predictionId: prediction.prediction_id || "",
            evaluatedAtUtc: prediction.evaluated_at_utc || new Date().toISOString(),
            label: prediction.label || "Unknown",
            confidence: typeof prediction.confidence === "number" ? prediction.confidence : 0,
            subject: payload.subject || "",
            sender: payload.sender || "",
            senderDomain: prediction.sender_domain || "",
            reason: prediction.reason || "",
            ruleLayer: prediction.rule_layer || "",
            userLabel: null,
            verdict: null
        };

        const filtered = history.filter((entry) => entry.predictionId !== prediction.prediction_id);
        filtered.unshift(nextEntry);
        await saveScanHistory(filtered.slice(0, settings.historyLimit));
    });
}

async function withHistoryLock(fn) {
    const prev = historyLock;
    let release;
    historyLock = new Promise((r) => { release = r; });
    try {
        await prev;
        return await fn();
    } catch (error) {
        console.error("History lock operation failed:", error);
        throw error;
    } finally {
        release();
    }
}

async function updateHistoryFeedback(predictionId, userLabel, verdict) {
    await withHistoryLock(async () => {
        const history = await getScanHistory();
        const updated = history.map((entry) => (
            entry.predictionId === predictionId
                ? { ...entry, userLabel, verdict }
                : entry
        ));
        await saveScanHistory(updated);
    });
}

async function analyzeEmail(payload) {
    const normalized = normalizePayload(payload);
    if (!normalized.subject && !normalized.body) {
        throw new Error("Email subject or body is required for analysis.");
    }

    const key = cacheKey(normalized);
    const cached = getCachedPrediction(key);
    if (cached) {
        return cached;
    }

    if (pendingAnalyses.has(key)) {
        return pendingAnalyses.get(key);
    }

    const analysisPromise = (async () => {
        try {
            const prediction = await fetchJson("/v1/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(normalized)
            });

            predictionCache.set(key, {
                timestamp: Date.now(),
                value: prediction
            });
            evictCacheIfNeeded();

            try {
                await pushHistoryEntry(normalized, prediction);
            } catch (error) {
                console.error("Failed to save history entry:", error);
            }

            return prediction;
        } finally {
            pendingAnalyses.delete(key);
        }
    })();

    pendingAnalyses.set(key, analysisPromise);
    return analysisPromise;
}

async function checkBackendHealth() {
    return fetchJson("/v1/health");
}

async function submitFeedback(payload) {
    if (!payload || typeof payload !== "object") {
        throw new Error("Feedback payload is required.");
    }
    const response = await fetchJson("/v1/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const verdict = response.verdict || "ok";
    try {
        await updateHistoryFeedback(payload.prediction_id, payload.user_label, verdict);
    } catch (error) {
        console.error("Failed to update local history after feedback:", error);
    }
    return response;
}

async function retrainModel() {
    const settings = await getSettings();
    predictionCache.clear();
    pendingAnalyses.clear();
    const response = await fetchJson("/v1/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    }, {
        ...settings,
        requestTimeoutMs: Math.max(settings.requestTimeoutMs, 15 * 60 * 1000)
    });
    return response;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const command = request?.command;

    if (command === "analyze_email") {
        analyzeEmail(request.payload)
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                console.error("Prediction request failed", error);
                try { sendResponse({ ok: false, error: error.message || "Prediction failed." }); } catch (e) {}
            });
        return true;
    }

    if (command === "check_backend_health") {
        checkBackendHealth()
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Backend is unavailable." }); } catch (e) {}
            });
        return true;
    }

    if (command === "get_settings") {
        getSettings()
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not load settings." }); } catch (e) {}
            });
        return true;
    }

    if (command === "save_settings") {
        saveSettings(request.payload)
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not save settings." }); } catch (e) {}
            });
        return true;
    }

    if (command === "get_scan_history") {
        getScanHistory()
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not load scan history." }); } catch (e) {}
            });
        return true;
    }

    if (command === "clear_scan_history") {
        withHistoryLock(async () => {
            await saveScanHistory([]);
        })
            .then(() => {
                try { sendResponse({ ok: true, data: [] }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not clear scan history." }); } catch (e) {}
            });
        return true;
    }

    if (command === "submit_feedback") {
        submitFeedback(request.payload)
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not submit feedback." }); } catch (e) {}
            });
        return true;
    }

    if (command === "retrain_model") {
        retrainModel()
            .then((data) => {
                try { sendResponse({ ok: true, data }); } catch (e) {}
            })
            .catch((error) => {
                try { sendResponse({ ok: false, error: error.message || "Could not retrain model." }); } catch (e) {}
            });
        return true;
    }

    if (command === "clear_prediction_cache") {
        predictionCache.clear();
        pendingAnalyses.clear();
        try { sendResponse({ ok: true, data: null }); } catch (e) {}
        return false;
    }

    return false;
});
