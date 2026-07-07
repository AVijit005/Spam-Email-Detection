const BANNER_ID = "spam-detector-banner";
const ANALYZE_DEBOUNCE_MS = 900;

let analyzeTimer = null;
let lastSignature = "";
let analysisInFlight = false;
let autoScanEnabled = true;
let cachedSettings = null;

async function getSettingsDirect() {
    if (cachedSettings) return cachedSettings;
    return new Promise((resolve) => {
        chrome.storage.sync.get("settings", (data) => {
            const stored = data.settings || {};
            cachedSettings = {
                apiBaseUrl: stored.apiBaseUrl || "https://avijit070-email-classifier-api.hf.space",
                apiKey: stored.apiKey || "",
                requestTimeoutMs: Math.max(25000, Number(stored.requestTimeoutMs) || 30000),
                autoScanEnabled: Boolean(stored.autoScanEnabled !== undefined ? stored.autoScanEnabled : true)
            };
            resolve(cachedSettings);
        });
    });
}

async function directFetch(path, options = {}) {
    const settings = await getSettingsDirect();
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
                ? body.detail : body || `Request failed with status ${response.status}`;
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

function runtimeMessage(message) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(message, (response) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
                return;
            }
            if (!response?.ok) {
                reject(new Error(response?.error || "Unknown extension error."));
                return;
            }
            resolve(response.data);
        });
    });
}

function normalizePayload(payload) {
    return {
        sender: String(payload.sender || "").trim(),
        subject: String(payload.subject || "").trim(),
        body: String(payload.body || "").trim()
    };
}

async function refreshSettings() {
    try {
        cachedSettings = null;
        const settings = await getSettingsDirect();
        autoScanEnabled = settings.autoScanEnabled;
    } catch (error) {
        autoScanEnabled = true;
    }
}

function emailSignature(data) {
    return JSON.stringify([data.sender || "", data.subject || "", data.body || ""]);
}

function removeBanner() {
    document.getElementById(BANNER_ID)?.remove();
}

function normalizedFeedbackLabel(predictionLabel) {
    return predictionLabel === "Spam" ? "Spam" : "Not Spam";
}

async function submitBannerFeedback(payload, prediction, userLabel, statusNode, actionButtons) {
    if (!payload || !prediction?.prediction_id) {
        statusNode.textContent = "Feedback is unavailable for this message.";
        statusNode.hidden = false;
        return;
    }

    actionButtons.forEach((button) => {
        button.disabled = true;
    });
    statusNode.hidden = false;
    statusNode.textContent = "Saving feedback...";

    try {
        const response = await directFetch("/v1/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prediction_id: prediction.prediction_id,
                sender: payload.sender || "",
                subject: payload.subject || "",
                body: payload.body || "",
                predicted_label: prediction.label,
                predicted_confidence: prediction.confidence,
                user_label: userLabel,
                source: "gmail_banner"
            })
        });
        statusNode.textContent = `Feedback saved (${String(response.verdict || "ok").replace("_", " ")}).`;
    } catch (error) {
        actionButtons.forEach((button) => {
            button.disabled = false;
        });
        statusNode.textContent = error.message || "Could not save feedback.";
    }
}

function createFeedbackButton(label, modifierClass, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `spam-detector-banner__action ${modifierClass}`;
    button.textContent = label;
    button.addEventListener("click", onClick);
    return button;
}

function createBanner(prediction, payload) {
    const banner = document.createElement("section");
    banner.id = BANNER_ID;

    const variant = prediction.label === "Spam"
        ? "spam"
        : prediction.label === "whitelisted"
            ? "trusted"
            : "safe";

    banner.className = `spam-detector-banner spam-detector-banner--${variant}`;

    const header = document.createElement("div");
    header.className = "spam-detector-banner__header";

    const badge = document.createElement("span");
    badge.className = "spam-detector-banner__badge";
    badge.textContent = prediction.label === "Spam"
        ? "Spam alert"
        : prediction.label === "whitelisted"
            ? "Whitelisted"
            : "Looks safe";

    const confidence = document.createElement("span");
    confidence.className = "spam-detector-banner__confidence";
    confidence.textContent = `${Math.round((prediction.confidence || 0) * 100)}% confidence`;

    header.append(badge, confidence);
    banner.appendChild(header);

    const reason = document.createElement("p");
    reason.className = "spam-detector-banner__reason";
    reason.textContent = prediction.reason || "Analysis completed.";
    banner.appendChild(reason);

    if (prediction.analysis) {
        const analysis = document.createElement("p");
        analysis.className = "spam-detector-banner__analysis";
        analysis.textContent = prediction.analysis;
        banner.appendChild(analysis);
    }

    const cues = (prediction.explanations?.length ? prediction.explanations : prediction.signals || []).slice(0, 3);
    if (cues.length) {
        const signals = document.createElement("div");
        signals.className = "spam-detector-banner__signals";
        cues.forEach((signal) => {
            const chip = document.createElement("span");
            chip.className = "spam-detector-banner__chip";
            chip.textContent = signal;
            signals.appendChild(chip);
        });
        banner.appendChild(signals);
    }

    const feedbackSection = document.createElement("div");
    feedbackSection.className = "spam-detector-banner__feedback";

    const feedbackLabel = document.createElement("span");
    feedbackLabel.className = "spam-detector-banner__feedback-label";
    feedbackLabel.textContent = "Feedback";

    const feedbackActions = document.createElement("div");
    feedbackActions.className = "spam-detector-banner__actions";

    const feedbackStatus = document.createElement("p");
    feedbackStatus.className = "spam-detector-banner__feedback-status";
    feedbackStatus.hidden = true;

    const buttons = [];
    const correctButton = createFeedbackButton(
        "Looks right",
        "spam-detector-banner__action--neutral",
        () => submitBannerFeedback(payload, prediction, normalizedFeedbackLabel(prediction.label), feedbackStatus, buttons),
    );
    const spamButton = createFeedbackButton(
        "Mark Spam",
        "spam-detector-banner__action--danger",
        () => submitBannerFeedback(payload, prediction, "Spam", feedbackStatus, buttons),
    );
    const safeButton = createFeedbackButton(
        "Mark Safe",
        "spam-detector-banner__action--safe",
        () => submitBannerFeedback(payload, prediction, "Not Spam", feedbackStatus, buttons),
    );

    buttons.push(correctButton, spamButton, safeButton);
    feedbackActions.append(correctButton, spamButton, safeButton);
    feedbackSection.append(feedbackLabel, feedbackActions, feedbackStatus);
    banner.appendChild(feedbackSection);

    return banner;
}

function injectBanner(prediction, payload) {
    removeBanner();
    const banner = createBanner(prediction, payload);

    let anchor = window.DomParser?.getBannerAnchor?.();
    if (!anchor) {
        const bodyEl = window.DomParser?.getBodyElement?.();
        if (bodyEl) {
            anchor = bodyEl.closest("[role="list"]") || bodyEl.closest(".nH") || bodyEl.parentElement?.parentElement;
        }
    }
    if (!anchor) {
        anchor = document.querySelector('[role="list"]') || document.querySelector(".nH");
    }
    if (anchor) {
        anchor.insertBefore(banner, anchor.firstChild);
    } else {
        console.warn("Spam Detector: could not find banner anchor in Gmail DOM");
    }
}

function injectWarningBanner(message) {
    removeBanner();
    const banner = document.createElement("section");
    banner.id = BANNER_ID;
    banner.className = "spam-detector-banner spam-detector-banner--warning";
    const text = document.createElement("p");
    text.className = "spam-detector-banner__reason";
    text.textContent = message;
    banner.appendChild(text);
    const anchor = window.DomParser?.getBannerAnchor?.();
    if (anchor) {
        anchor.insertBefore(banner, anchor.firstChild);
    }
}

function injectLoadingBanner() {
    removeBanner();
    const banner = document.createElement("section");
    banner.id = BANNER_ID;
    banner.className = "spam-detector-banner spam-detector-banner--safe";
    const text = document.createElement("p");
    text.className = "spam-detector-banner__reason";
    text.textContent = "Analyzing this email...";
    banner.appendChild(text);
    const anchor = window.DomParser?.getBannerAnchor?.();
    if (anchor) {
        anchor.insertBefore(banner, anchor.firstChild);
    }
}

let retryCount = 0;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 1500;

async function analyzeOpenEmail(force = false) {
    const data = window.DomParser?.getEmailData?.();
    if (!data || (!data.subject && !data.body)) {
        if (force && retryCount < MAX_RETRIES) {
            retryCount++;
            setTimeout(() => analyzeOpenEmail(true), RETRY_DELAY_MS);
        }
        return;
    }

    if (!data.sender && !data.subject && !data.body) {
        injectWarningBanner("Spam Detector could not read this email — Gmail may have updated its layout. Try refreshing or pasting into the popup.");
        return;
    }

    if (!autoScanEnabled && !force) {
        return;
    }

    const signature = emailSignature(data);
    if (!force && (analysisInFlight || signature === lastSignature)) {
        return;
    }

    retryCount = 0;
    analysisInFlight = true;
    lastSignature = signature;

    injectLoadingBanner();

    try {
        const normalized = normalizePayload(data);
        const prediction = await directFetch("/v1/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(normalized)
        });
        injectBanner(prediction, data);
    } catch (error) {
        lastSignature = "";
        console.warn("Spam detector could not analyze email:", error.message);
    } finally {
        analysisInFlight = false;
    }
}

function scheduleAnalysis(force = false) {
    clearTimeout(analyzeTimer);
    analyzeTimer = setTimeout(() => analyzeOpenEmail(force), ANALYZE_DEBOUNCE_MS);
}

const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
            scheduleAnalysis(false);
            break;
        }
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

document.addEventListener("visibilitychange", async () => {
    if (!document.hidden) {
        await refreshSettings();
        lastSignature = "";
        scheduleAnalysis(true);
    }
});

window.addEventListener("hashchange", () => {
    lastSignature = "";
    scheduleAnalysis(true);
});

let pollTimer = null;
function startPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(() => {
        if (!document.hidden && autoScanEnabled) {
            const data = window.DomParser?.getEmailData?.();
            if (data && (data.subject || data.body)) {
                const sig = emailSignature(data);
                if (sig !== lastSignature && !analysisInFlight) {
                    scheduleAnalysis(true);
                }
            }
        }
    }, 5000);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.command === "get_email_data") {
        sendResponse(window.DomParser?.getEmailData?.() || null);
    }
});

refreshSettings().then(() => {
    scheduleAnalysis(true);
    startPolling();
});
