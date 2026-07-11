const DEFAULT_SETTINGS = {
    apiBaseUrl: "https://avijit070-email-classifier-api.hf.space",
    apiKey: "",
    requestTimeoutMs: 30000,
    autoScanEnabled: true,
    historyLimit: 12
};

const CACHE_TTL_MS = 90 * 1000;

chrome.runtime.onInstalled.addListener(async () => {
    try {
        const current = await getSettings();
        const normalized = { ...current, apiBaseUrl: normalizeApiBaseUrl(current.apiBaseUrl) };
        await chrome.storage.sync.set({ settings: normalized });
    } catch (e) {
        const settings = { ...DEFAULT_SETTINGS };
        try {
            await chrome.storage.sync.set({ settings });
        } catch (_) { /* ignore */ }
    }
    console.log("Gmail Spam Detector extension installed");
});

async function getCachedPrediction(key) {
    try {
        const data = await chrome.storage.session.get(key);
        const cached = data[key];
        if (!cached) return null;
        if ((Date.now() - cached.timestamp) > CACHE_TTL_MS) {
            chrome.storage.session.remove(key);
            return null;
        }
        return cached.value;
    } catch (e) {
        return null;
    }
}

async function setCachedPrediction(key, value) {
    try {
        await chrome.storage.session.set({ [key]: { timestamp: Date.now(), value } });
    } catch (e) {
        /* ignore quota / unavailable errors */
    }
}

function normalizePayload(payload = {}) {
    return {
        sender: typeof payload.sender === "string" ? payload.sender.trim() : "",
        subject: typeof payload.subject === "string" ? payload.subject.trim() : "",
        body: typeof payload.body === "string" ? payload.body.trim() : ""
    };
}

function cacheKey(payload) {
    return JSON.stringify([payload.sender, payload.subject, payload.body]);
}

async function getSettings() {
    const data = await chrome.storage.sync.get("settings");
    const stored = data.settings || {};
    return {
        ...DEFAULT_SETTINGS,
        ...stored
    };
}

function normalizeApiBaseUrl(url) {
    const value = String(url || "").trim().replace(/\/+$/, "");
    const normalized = value || DEFAULT_SETTINGS.apiBaseUrl;
    const parsed = new URL(normalized);

    if (parsed.protocol === "https:") {
        return parsed.origin;
    }

    if (parsed.protocol === "http:" && ["localhost", "127.0.0.1"].includes(parsed.hostname)) {
        return parsed.origin;
    }

    throw new Error("Use http:// only for localhost, or https:// for deployed backends.");
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
        const body = contentType.includes("application/json")
            ? await response.json()
            : await response.text();

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
    const settings = await getSettings();
    const history = await getScanHistory();
    const nextEntry = {
        predictionId: prediction.prediction_id,
        evaluatedAtUtc: prediction.evaluated_at_utc,
        label: prediction.label,
        confidence: prediction.confidence,
        subject: payload.subject,
        sender: payload.sender,
        senderDomain: prediction.sender_domain,
        reason: prediction.reason,
        ruleLayer: prediction.rule_layer,
        userLabel: null,
        verdict: null
    };

    const filtered = history.filter((entry) => entry.predictionId !== prediction.prediction_id);
    filtered.unshift(nextEntry);
    await saveScanHistory(filtered.slice(0, settings.historyLimit));
}

async function updateHistoryFeedback(predictionId, userLabel, verdict) {
    const history = await getScanHistory();
    const updated = history.map((entry) => (
        entry.predictionId === predictionId
            ? { ...entry, userLabel, verdict }
            : entry
    ));
    await saveScanHistory(updated);
}

async function analyzeEmail(payload) {
    const normalized = normalizePayload(payload);
    if (!normalized.subject && !normalized.body) {
        throw new Error("Email subject or body is required for analysis.");
    }

    const key = cacheKey(normalized);
    const cached = await getCachedPrediction(key);
    if (cached) {
        return cached;
    }

    const prediction = await fetchJson("/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(normalized)
    });

    await setCachedPrediction(key, prediction);
    await pushHistoryEntry(normalized, prediction);
    return prediction;
}

async function checkBackendHealth() {
    return fetchJson("/v1/health");
}

async function submitFeedback(payload) {
    const response = await fetchJson("/v1/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    await updateHistoryFeedback(payload.prediction_id, payload.user_label, response.verdict);
    return response;
}

async function retrainModel() {
    const settings = await getSettings();
    const response = await fetchJson("/v1/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    }, {
        ...settings,
        requestTimeoutMs: Math.max(settings.requestTimeoutMs, 15 * 60 * 1000)
    });
    try {
        await chrome.storage.session.clear();
    } catch (e) { /* ignore */ }
    return response;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const command = request?.command;

    if (command === "analyze_email") {
        analyzeEmail(request.payload)
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => {
                console.error("Prediction request failed", error);
                sendResponse({ ok: false, error: error.message || "Prediction failed." });
            });
        return true;
    }

    if (command === "check_backend_health") {
        checkBackendHealth()
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Backend is unavailable." }));
        return true;
    }

    if (command === "get_settings") {
        getSettings()
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not load settings." }));
        return true;
    }

    if (command === "save_settings") {
        saveSettings(request.payload)
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not save settings." }));
        return true;
    }

    if (command === "get_scan_history") {
        getScanHistory()
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not load scan history." }));
        return true;
    }

    if (command === "clear_scan_history") {
        saveScanHistory([])
            .then(() => sendResponse({ ok: true, data: [] }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not clear scan history." }));
        return true;
    }

    if (command === "submit_feedback") {
        submitFeedback(request.payload)
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not submit feedback." }));
        return true;
    }

    if (command === "retrain_model") {
        retrainModel()
            .then((data) => sendResponse({ ok: true, data }))
            .catch((error) => sendResponse({ ok: false, error: error.message || "Could not retrain model." }));
        return true;
    }

    if (command === "clear_prediction_cache") {
        chrome.storage.session.clear()
            .then(() => sendResponse({ ok: true }))
            .catch(() => sendResponse({ ok: true }));
        return true;
    }

    return false;
});
