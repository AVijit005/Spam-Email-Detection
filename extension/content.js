const BANNER_ID = "spam-detector-banner";
const ANALYZE_DEBOUNCE_MS = 900;
const MESSAGE_TIMEOUT_MS = 60000;

let analyzeTimer = null;
let lastSignature = "";
let analysisInFlight = false;
let autoScanEnabled = true;
let pendingForce = false;
let feedbackInFlight = false;
let confidenceBarTimer = null;

function runtimeMessage(message) {
    return new Promise((resolve, reject) => {
        let settled = false;
        const timer = setTimeout(() => {
            if (!settled) {
                settled = true;
                reject(new Error("Extension message timed out."));
            }
        }, MESSAGE_TIMEOUT_MS);

        chrome.runtime.sendMessage(message, (response) => {
            clearTimeout(timer);
            if (settled) return;
            settled = true;
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

async function refreshSettings() {
    try {
        const settings = await runtimeMessage({ command: "get_settings" });
        autoScanEnabled = Boolean(settings.autoScanEnabled);
    } catch (error) {
        autoScanEnabled = true;
    }
}

function readCurrentEmailData() {
    return window.DomParser?.getEmailData?.() || null;
}

function emailSignature(data) {
    if (!data) return "";
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

    if (feedbackInFlight) return;
    feedbackInFlight = true;

    actionButtons.forEach((button) => {
        button.disabled = true;
    });
    statusNode.hidden = false;
    statusNode.textContent = "Saving feedback...";

    try {
        const response = await runtimeMessage({
            command: "submit_feedback",
            payload: {
                prediction_id: prediction.prediction_id,
                sender: payload.sender || "",
                subject: payload.subject || "",
                body: payload.body || "",
                predicted_label: prediction.label,
                predicted_confidence: prediction.confidence,
                user_label: userLabel,
                source: "gmail_banner"
            }
        });
        statusNode.textContent = `Feedback saved (${String(response?.verdict || "ok").replace(/_/g, " ")}).`;
    } catch (error) {
        actionButtons.forEach((button) => {
            button.disabled = false;
        });
        statusNode.textContent = error.message || "Could not save feedback.";
    } finally {
        feedbackInFlight = false;
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
    if (!prediction || typeof prediction !== "object") {
        return null;
    }

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
        ? "Spam"
        : prediction.label === "whitelisted"
            ? "Whitelisted"
            : "Safe";

    const confidence = document.createElement("span");
    confidence.className = "spam-detector-banner__confidence";
    confidence.textContent = `${Math.round((prediction.confidence || 0) * 100)}%`;

    const expandIcon = document.createElement("span");
    expandIcon.className = "spam-detector-banner__expand-icon";
    expandIcon.textContent = "\u25B6";

    header.append(badge, confidence, expandIcon);
    banner.appendChild(header);

    const details = document.createElement("div");
    details.className = "spam-detector-banner__details";

    const reason = document.createElement("p");
    reason.className = "spam-detector-banner__reason";
    reason.textContent = prediction.reason || "Analysis completed.";
    details.appendChild(reason);

    if (prediction.analysis) {
        const analysis = document.createElement("p");
        analysis.className = "spam-detector-banner__analysis";
        analysis.textContent = prediction.analysis;
        details.appendChild(analysis);
    }

    const rawCues = prediction.explanations?.length
        ? prediction.explanations
        : Array.isArray(prediction.signals) ? prediction.signals : [];
    const cues = rawCues.slice(0, 3);
    if (cues.length) {
        const signals = document.createElement("div");
        signals.className = "spam-detector-banner__signals";
        cues.forEach((signal) => {
            if (signal == null) return;
            const chip = document.createElement("span");
            chip.className = "spam-detector-banner__chip";
            chip.textContent = String(signal);
            signals.appendChild(chip);
        });
        details.appendChild(signals);
    }

    const feedbackSection = document.createElement("div");
    feedbackSection.className = "spam-detector-banner__feedback";

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
    feedbackSection.append(feedbackActions, feedbackStatus);
    details.appendChild(feedbackSection);
    banner.appendChild(details);

    header.addEventListener("click", (event) => {
        event.stopPropagation();
        banner.classList.toggle("spam-detector-banner--expanded");
    });

    return banner;
}

function injectBanner(prediction, payload) {
    removeBanner();
    const banner = createBanner(prediction, payload);
    if (!banner) return;
    const anchor = window.DomParser?.getBannerAnchor?.();
    if (!anchor) return;
    anchor.insertBefore(banner, anchor.firstChild);
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

async function analyzeOpenEmail(force = false) {
    const data = readCurrentEmailData();
    if (!data || (!data.subject && !data.body)) {
        return;
    }

    if (!autoScanEnabled && !force) {
        return;
    }

    const currentSignature = emailSignature(data);
    if (!force && (analysisInFlight || currentSignature === lastSignature)) {
        return;
    }

    analysisInFlight = true;
    lastSignature = currentSignature;
    const analysisSignature = currentSignature;

    try {
        const prediction = await runtimeMessage({
            command: "analyze_email",
            payload: data
        });
        const freshData = readCurrentEmailData();
        if (emailSignature(freshData) === analysisSignature) {
            injectBanner(prediction, freshData || data);
        }
    } catch (error) {
        const freshData = readCurrentEmailData();
        if (emailSignature(freshData) === analysisSignature) {
            console.warn("Spam detector could not analyze email:", error.message);
        }
    } finally {
        analysisInFlight = false;
    }
}

function scheduleAnalysis(force = false) {
    if (force) {
        pendingForce = true;
    }
    clearTimeout(analyzeTimer);
    analyzeTimer = setTimeout(() => {
        const shouldForce = pendingForce;
        pendingForce = false;
        analyzeOpenEmail(shouldForce);
    }, ANALYZE_DEBOUNCE_MS);
}

function setupObserver() {
    if (!document.body) return;
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.addedNodes.length > 0) {
                scheduleAnalysis(false);
                break;
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

if (document.body) {
    setupObserver();
} else {
    document.addEventListener("DOMContentLoaded", setupObserver);
}

document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
        refreshSettings().then(() => scheduleAnalysis(true));
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request?.command === "get_email_data") {
        sendResponse(readCurrentEmailData());
    }
});

refreshSettings().then(() => {
    scheduleAnalysis(true);
}).catch(() => {
    scheduleAnalysis(true);
});
