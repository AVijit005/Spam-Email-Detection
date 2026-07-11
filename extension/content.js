const BANNER_ID = "spam-detector-banner";
const ANALYZE_DEBOUNCE_MS = 900;

console.log("[SpamDetector] content script loaded");

const FALLBACK_SETTINGS = {
    apiBaseUrl: "https://avijit070-email-classifier-api.hf.space",
    apiKey: "",
    requestTimeoutMs: 30000,
    autoScanEnabled: true
};

let analyzeTimer = null;
let lastSignature = "";
let analysisInFlight = false;
let autoScanEnabled = true;
let cachedSettings = null;
let extensionAlive = true;
let settingsListenerAttached = false;

function isExtensionContextValid() {
    try {
        return typeof chrome !== "undefined" && chrome.runtime && !!chrome.runtime.id;
    } catch (e) {
        return false;
    }
}

async function getSettingsDirect() {
    if (cachedSettings) return cachedSettings;
    if (!isExtensionContextValid()) {
        extensionAlive = false;
        cachedSettings = { ...FALLBACK_SETTINGS };
        return cachedSettings;
    }
    try {
        const data = await new Promise((resolve) => {
            chrome.storage.sync.get("settings", (result) => resolve(result || {}));
        });
        const stored = data.settings || {};
        cachedSettings = {
            apiBaseUrl: stored.apiBaseUrl || FALLBACK_SETTINGS.apiBaseUrl,
            apiKey: stored.apiKey || FALLBACK_SETTINGS.apiKey,
            requestTimeoutMs: Number(stored.requestTimeoutMs) || FALLBACK_SETTINGS.requestTimeoutMs,
            autoScanEnabled: stored.autoScanEnabled !== undefined ? Boolean(stored.autoScanEnabled) : FALLBACK_SETTINGS.autoScanEnabled
        };
    } catch (e) {
        extensionAlive = false;
        cachedSettings = { ...FALLBACK_SETTINGS };
    }
    return cachedSettings;
}

function analyzeViaBackground(payload, attempt = 1) {
    return new Promise((resolve, reject) => {
        if (!isExtensionContextValid()) {
            extensionAlive = false;
            reject(new Error("Extension context invalidated. Refresh Gmail to re-enable auto-scan."));
            return;
        }
        chrome.runtime.sendMessage({ command: "analyze_email", payload }, (response) => {
            if (chrome.runtime.lastError) {
                const message = chrome.runtime.lastError.message || "";
                if (attempt < 2 && /Receiving end does not exist|message port closed/i.test(message)) {
                    setTimeout(() => analyzeViaBackground(payload, attempt + 1).then(resolve, reject), 400);
                    return;
                }
                reject(new Error(message));
                return;
            }
            if (!response?.ok) {
                reject(new Error(response?.error || "Prediction failed."));
                return;
            }
            resolve(response.data);
        });
    });
}

function submitFeedbackViaBackground(payload) {
    return new Promise((resolve, reject) => {
        if (!isExtensionContextValid()) {
            reject(new Error("Extension context invalidated."));
            return;
        }
        chrome.runtime.sendMessage({ command: "submit_feedback", payload }, (response) => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
                return;
            }
            if (!response?.ok) {
                reject(new Error(response?.error || "Feedback failed."));
                return;
            }
            resolve(response.data);
        });
    });
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

function stopAutoScan() {
    clearTimeout(analyzeTimer);
    clearInterval(pollTimer);
    if (observer) observer.disconnect();
}

function emailSignature(data) {
    return JSON.stringify([data.sender || "", data.subject || "", data.body || ""]);
}

function removeBanner() {
    document.getElementById(BANNER_ID)?.remove();
    const host = document.getElementById("spam-detector-host");
    if (host) host.innerHTML = "";
}

const BANNER_HOST_ID = "spam-detector-host";
function getBannerHost() {
    let host = document.getElementById(BANNER_HOST_ID);
    if (!host) {
        host = document.createElement("div");
        host.id = BANNER_HOST_ID;
        host.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:2147483647;width:min(440px,94vw);pointer-events:none;";
        (document.body || document.documentElement).appendChild(host);
    }
    return host;
}

function mountBanner(banner) {
    const host = getBannerHost();
    host.innerHTML = "";
    banner.style.position = "relative";
    banner.style.pointerEvents = "auto";
    banner.style.width = "100%";
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "×";
    close.setAttribute("aria-label", "Dismiss");
    close.style.cssText = "position:absolute;top:4px;right:6px;width:22px;height:22px;border:0;border-radius:6px;background:rgba(0,0,0,0.28);color:inherit;font-size:16px;line-height:1;cursor:pointer;opacity:0.7;";
    close.addEventListener("click", () => { host.innerHTML = ""; });
    banner.appendChild(close);
    host.appendChild(banner);
}

function findBannerAnchor() {
    let anchor = window.DomParser?.getBannerAnchor?.();
    if (anchor) return anchor;
    const bodyEl = window.DomParser?.getBodyElement?.();
    if (bodyEl) {
        anchor = bodyEl.closest(".nH") || bodyEl.closest("[role='main']") || bodyEl.parentElement?.parentElement;
        if (anchor) return anchor;
    }
    return document.querySelector("[role='main']") || document.querySelector(".nH") || null;
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
        const response = await submitFeedbackViaBackground({
            prediction_id: prediction.prediction_id,
            sender: payload.sender || "",
            subject: payload.subject || "",
            body: payload.body || "",
            predicted_label: prediction.label,
            predicted_confidence: prediction.confidence,
            user_label: userLabel,
            source: "gmail_banner"
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

const ICONS = {
    spam: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    safe: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    trusted: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    warning: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
};

function buildBanner(variant, badgeText, confidenceText) {
    const section = document.createElement("section");
    section.id = BANNER_ID;
    section.className = `spam-detector-banner spam-detector-banner--${variant}`;

    const header = document.createElement("div");
    header.className = "spam-detector-banner__header";

    const icon = document.createElement("span");
    icon.className = "spam-detector-banner__icon";
    icon.innerHTML = ICONS[variant] || ICONS.warning;

    const badge = document.createElement("span");
    badge.className = "spam-detector-banner__badge";
    badge.textContent = badgeText;

    header.append(icon, badge);

    if (confidenceText) {
        const confidence = document.createElement("span");
        confidence.className = "spam-detector-banner__confidence";
        confidence.textContent = confidenceText;
        header.append(confidence);
    }

    section.append(header);
    return section;
}

function createBanner(prediction, payload) {
    const isWhitelisted = prediction.label === "whitelisted";
    const isTrustedService = prediction.label === "Not Spam" && prediction.rule_layer === "trusted_service";
    const variant = prediction.label === "Spam"
        ? "spam"
        : (isWhitelisted || isTrustedService)
            ? "trusted"
            : "safe";

    const badgeText = isWhitelisted
        ? "Whitelisted sender"
        : isTrustedService
            ? "Trusted sender"
            : prediction.label === "Spam"
                ? "Spam alert"
                : "Looks safe";

    const banner = buildBanner(variant, badgeText, `${Math.round((prediction.confidence || 0) * 100)}% confidence`);

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
    feedbackLabel.textContent = "Was this correct?";

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
    mountBanner(createBanner(prediction, payload));
}

function showContextBanner() {
    const banner = buildBanner("warning", "Action needed");
    const text = document.createElement("p");
    text.className = "spam-detector-banner__reason";
    text.textContent = "Spam Detector was updated — hard-refresh this Gmail tab (Ctrl+Shift+R) to enable auto-scan.";
    banner.appendChild(text);
    mountBanner(banner);
}

function injectWarningBanner(message) {
    const banner = buildBanner("warning", "Heads up");
    const text = document.createElement("p");
    text.className = "spam-detector-banner__reason";
    text.textContent = message;
    banner.appendChild(text);
    mountBanner(banner);
}

function injectLoadingBanner() {
    const banner = buildBanner("safe", "Analyzing…");
    const text = document.createElement("p");
    text.className = "spam-detector-banner__reason";
    text.textContent = "Scanning this email for spam and phishing signals…";
    banner.appendChild(text);
    mountBanner(banner);
}

let retryCount = 0;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 1500;
let missedSignature = null;
let analysisSucceeded = false;

async function analyzeOpenEmail(force = false) {
    if (!extensionAlive) {
        showContextBanner();
        stopAutoScan();
        return;
    }
    let analysisErrored = false;
    const data = window.DomParser?.getEmailData?.();
    const hasEmail = !!(data && (data.subject || data.body));
    if (!hasEmail) {
        if (force && retryCount < MAX_RETRIES) {
            retryCount++;
            setTimeout(() => analyzeOpenEmail(true), RETRY_DELAY_MS);
            return;
        }
        retryCount = 0;
        removeBanner();
        return;
    }

    if (!autoScanEnabled && !force) {
        return;
    }

    const signature = emailSignature(data);
    if (!force && (analysisInFlight || signature === lastSignature)) {
        if (analysisInFlight) missedSignature = signature;
        return;
    }

    retryCount = 0;
    missedSignature = null;
    analysisSucceeded = false;
    analysisInFlight = true;
    lastSignature = signature;

    injectLoadingBanner();

    try {
        const normalized = {
            sender: String(data.sender || "").trim(),
            subject: String(data.subject || "").trim(),
            body: String(data.body || "").trim()
        };
        const prediction = await analyzeViaBackground(normalized);
        injectBanner(prediction, data);
        analysisSucceeded = true;
        console.log(`[SpamDetector] analyzed -> ${prediction.label} (${Math.round((prediction.confidence || 0) * 100)}%)`);
    } catch (error) {
        lastSignature = "";
        analysisErrored = true;
        console.warn("Spam detector could not analyze email:", error.message);
        injectWarningBanner(error.message || "Could not analyze this email.");
    } finally {
        if (!analysisSucceeded && !analysisErrored) {
            removeBanner();
        }
        analysisInFlight = false;
        const currentSig = emailSignature(window.DomParser?.getEmailData?.() || {});
        if (missedSignature && missedSignature !== currentSig) {
            missedSignature = null;
            setTimeout(() => scheduleAnalysis(false), 100);
        }
    }
}

function scheduleAnalysis(force = false) {
    clearTimeout(analyzeTimer);
    analyzeTimer = setTimeout(() => analyzeOpenEmail(force), ANALYZE_DEBOUNCE_MS);
}

const observer = new MutationObserver((mutations) => {
    if (!extensionAlive) return;
    const banner = document.getElementById(BANNER_ID);
    for (const mutation of mutations) {
        if (mutation.addedNodes.length === 0) continue;
        let node = mutation.target;
        if (banner && (node === banner || banner.contains(node))) continue;
        scheduleAnalysis(false);
        return;
    }
});

function observeEmailPane() {
    const pane = document.querySelector("[role='main']") || document.body;
    observer.observe(pane, {
        childList: true,
        subtree: true
    });
}

document.addEventListener("visibilitychange", async () => {
    if (!document.hidden && extensionAlive) {
        await refreshSettings();
        lastSignature = "";
        scheduleAnalysis(true);
    }
});

function onNavigate() {
    if (!extensionAlive) return;
    retryCount = 0;
    missedSignature = null;
    lastSignature = "";
    removeBanner();
    scheduleAnalysis(true);
}

window.addEventListener("hashchange", onNavigate);

let pollTimer = null;
function startPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(() => {
        if (!extensionAlive || document.hidden || !autoScanEnabled) return;
        const data = window.DomParser?.getEmailData?.();
        if (data && (data.subject || data.body)) {
            const sig = emailSignature(data);
            if (sig !== lastSignature && !analysisInFlight) {
                scheduleAnalysis(true);
            }
        }
    }, 5000);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!isExtensionContextValid()) {
        sendResponse(null);
        return;
    }
    if (request.command === "get_email_data") {
        sendResponse(window.DomParser?.getEmailData?.() || null);
        return;
    }
    if (request.command === "refresh_settings") {
        refreshSettings().then(() => sendResponse({ ok: true }));
        return true;
    }
    sendResponse(null);
});

function attachSettingsListener() {
    if (settingsListenerAttached) return;
    if (!isExtensionContextValid() || !chrome.storage?.onChanged) return;
    settingsListenerAttached = true;
    chrome.storage.onChanged.addListener((changes, area) => {
        if (area !== "sync" || !changes.settings) return;
        cachedSettings = null;
        refreshSettings().then(() => {
            if (autoScanEnabled) {
                lastSignature = "";
                scheduleAnalysis(true);
            }
        });
    });
}

refreshSettings().then(() => {
    if (!extensionAlive) {
        console.warn("[SpamDetector] Extension context unavailable. Refresh Gmail to re-enable auto-scan.");
        return;
    }
    attachSettingsListener();
    observeEmailPane();
    scheduleAnalysis(true);
    startPolling();
    console.log("[SpamDetector] auto-scan armed on", location.href);
});
