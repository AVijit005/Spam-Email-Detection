document.addEventListener("DOMContentLoaded", () => {
    const elements = {
        btnGet: document.getElementById("btn-get"),
        btnAnalyze: document.getElementById("btn-analyze"),
        btnSettings: document.getElementById("btn-settings"),
        btnClearHistory: document.getElementById("btn-clear-history"),
        btnFeedbackCorrect: document.getElementById("btn-feedback-correct"),
        btnFeedbackSpam: document.getElementById("btn-feedback-spam"),
        btnFeedbackSafe: document.getElementById("btn-feedback-safe"),
        emailInput: document.getElementById("email-input"),
        resultBox: document.getElementById("result"),
        resultContent: document.getElementById("result-content"),
        loaderContainer: document.querySelector(".loader-container"),
        label: document.getElementById("label"),
        confidence: document.getElementById("confidence"),
        reason: document.getElementById("reason"),
        analysis: document.getElementById("analysis"),
        confidenceFill: document.getElementById("confidence-fill"),
        resultMeta: document.getElementById("result-meta"),
        explanationsSection: document.getElementById("explanations-section"),
        explanationsList: document.getElementById("explanations-list"),
        feedbackSection: document.getElementById("feedback-section"),
        feedbackStatus: document.getElementById("feedback-status"),
        serviceStatus: document.getElementById("service-status"),
        serviceStatusText: document.getElementById("service-status-text"),
        healthMeta: document.getElementById("health-meta"),
        historyList: document.getElementById("history-list")
    };

    const missingElements = Object.entries(elements).filter(([, el]) => !el);
    if (missingElements.length) {
        console.warn("Spam detector: missing DOM elements:", missingElements.map(([k]) => k).join(", "));
    }

    let currentPayload = null;
    let currentPrediction = null;
    let confidenceBarTimer = null;
    let historyLoadCount = 0;
    let isLoadingFromGmail = false;
    let feedbackInFlight = false;
    let isAnalyzing = false;

    function runtimeMessage(message) {
        return new Promise((resolve, reject) => {
            let settled = false;
            const timer = setTimeout(() => {
                if (!settled) {
                    settled = true;
                    reject(new Error("Extension message timed out."));
                }
            }, 60000);

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

                resolve(response.data ?? null);
            });
        });
    }

    function resetResultState() {
        if (confidenceBarTimer) {
            clearTimeout(confidenceBarTimer);
            confidenceBarTimer = null;
        }
        elements.resultBox.classList.remove("visible", "spam", "safe", "whitelisted", "error");
        elements.resultBox.classList.add("hidden");
        elements.resultContent.classList.add("hidden");
        elements.loaderContainer.classList.add("hidden");
        elements.resultMeta.classList.add("hidden");
        elements.resultMeta.innerHTML = "";
        elements.explanationsSection.classList.add("hidden");
        elements.explanationsList.innerHTML = "";
        elements.feedbackSection.classList.add("hidden");
        elements.feedbackStatus.classList.add("hidden");
        elements.feedbackStatus.textContent = "";
        elements.confidenceFill.style.width = "0%";
        elements.label.textContent = "";
        elements.confidence.textContent = "";
        elements.reason.textContent = "";
        elements.analysis.textContent = "";
    }

    function setLoadingState(isLoading) {
        elements.btnGet.disabled = isLoading;
        elements.btnAnalyze.disabled = isLoading;
        if (isLoading) {
            resetResultState();
            elements.resultBox.classList.remove("hidden");
            elements.resultBox.classList.add("visible");
            elements.loaderContainer.classList.remove("hidden");
        } else {
            elements.loaderContainer.classList.add("hidden");
        }
    }

    function updateServiceStatus(text, isOnline) {
        elements.serviceStatus.classList.toggle("offline", !isOnline);
        elements.serviceStatusText.textContent = text;
    }

    function formatDate(isoString) {
        if (!isoString) return "";
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return "";
        return date.toLocaleString([], {
            hour: "2-digit",
            minute: "2-digit",
            month: "short",
            day: "numeric"
        });
    }

    async function refreshBackendStatus() {
        try {
            const health = await runtimeMessage({ command: "check_backend_health" });
            if (!health || typeof health !== "object") {
                updateServiceStatus("Backend offline", false);
                elements.healthMeta.classList.add("hidden");
                return;
            }
            const version = health.model_version && health.model_version !== "untrained"
                ? ` \u2022 ${health.model_version}`
                : "";
            updateServiceStatus(`Backend online${version}`, true);

            const metaBits = [
                health.trained_at_utc ? `Trained ${formatDate(health.trained_at_utc)}` : null,
                health.feedback_count != null ? `Feedback ${health.feedback_count}` : null,
                health.user_whitelist_count != null ? `Whitelist ${health.user_whitelist_count}` : null
            ].filter(Boolean);

            elements.healthMeta.innerHTML = "";
            metaBits.forEach((bit) => {
                const chip = document.createElement("span");
                chip.className = "meta-chip";
                chip.textContent = bit;
                elements.healthMeta.appendChild(chip);
            });
            elements.healthMeta.classList.toggle("hidden", metaBits.length === 0);
        } catch (error) {
            updateServiceStatus("Backend offline", false);
            elements.healthMeta.classList.add("hidden");
        }
    }

    function parseEmail(text) {
        const lines = text.replace(/\r\n/g, "\n").split("\n");
        let sender = "";
        let subject = "";
        let bodyStartIndex = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const senderMatch = line.match(/^\s*From:\s*(.+)/i);
            const subjectMatch = line.match(/^\s*Subject:\s*(.+)/i);

            if (senderMatch && !sender) {
                sender = senderMatch[1].trim();
                bodyStartIndex = i + 1;
            } else if (subjectMatch && !subject) {
                subject = subjectMatch[1].trim();
                bodyStartIndex = i + 1;
            } else if (line.trim() === "") {
                bodyStartIndex = i + 1;
                break;
            } else {
                bodyStartIndex = i + 1;
            }
        }

        const body = lines.slice(bodyStartIndex).join("\n").trim();
        return { sender, subject, body };
    }

    function addMetaChip(text) {
        const chip = document.createElement("span");
        chip.className = "meta-chip";
        chip.textContent = text;
        elements.resultMeta.appendChild(chip);
    }

    function renderExplanations(explanations = []) {
        elements.explanationsList.innerHTML = "";
        if (!Array.isArray(explanations) || explanations.length === 0) {
            elements.explanationsSection.classList.add("hidden");
            return;
        }

        explanations.slice(0, 4).forEach((entry) => {
            const item = document.createElement("li");
            item.textContent = typeof entry === "string" ? entry : String(entry);
            elements.explanationsList.appendChild(item);
        });
        elements.explanationsSection.classList.remove("hidden");
    }

    function renderFeedbackSection() {
        if (!currentPrediction || !currentPayload) {
            elements.feedbackSection.classList.add("hidden");
            return;
        }

        elements.feedbackSection.classList.remove("hidden");
        if (!feedbackInFlight) {
            elements.btnFeedbackCorrect.disabled = false;
            elements.btnFeedbackSpam.disabled = false;
            elements.btnFeedbackSafe.disabled = false;
        }
    }

    function renderResult(data, payload) {
        if (!data || typeof data !== "object") {
            renderError("Invalid response from backend.");
            return;
        }

        currentPrediction = data;
        currentPayload = payload;

        resetResultState();
        elements.resultBox.classList.remove("hidden");
        elements.resultBox.classList.add("visible");
        elements.resultContent.classList.remove("hidden");

        const validLabel = typeof data.label === "string" ? data.label : "";
        const cssClass = validLabel === "Spam"
            ? "spam"
            : validLabel === "whitelisted"
                ? "whitelisted"
                : "safe";

        const displayLabel = validLabel === "whitelisted"
            ? "WHITELISTED"
            : (validLabel || "UNKNOWN").toUpperCase();

        elements.resultBox.classList.add(cssClass);
        elements.label.textContent = displayLabel;
        elements.confidence.textContent = `Confidence: ${Math.round((data.confidence || 0) * 100)}%`;
        elements.reason.textContent = data.reason || "";
        elements.analysis.textContent = data.analysis || "";

        if (confidenceBarTimer) clearTimeout(confidenceBarTimer);
        confidenceBarTimer = setTimeout(() => {
            elements.confidenceFill.style.width = `${Math.round((data.confidence || 0) * 100)}%`;
        }, 80);

        if (data.rule_layer) addMetaChip(`Layer: ${data.rule_layer}`);
        if (data.model_version) addMetaChip(data.model_version);
        if (data.sender_domain) addMetaChip(data.sender_domain);
        if (data.evaluated_at_utc) addMetaChip(formatDate(data.evaluated_at_utc));
        if (Array.isArray(data.signals) && data.signals.length) {
            data.signals.slice(0, 3).forEach((s) => addMetaChip(String(s)));
        }

        if (elements.resultMeta.childElementCount > 0) {
            elements.resultMeta.classList.remove("hidden");
        }

        renderExplanations(data.explanations || []);
        renderFeedbackSection();
    }

    function renderError(message) {
        currentPrediction = null;
        currentPayload = null;
        resetResultState();
        elements.resultBox.classList.remove("hidden");
        elements.resultBox.classList.add("visible", "error");
        elements.resultContent.classList.remove("hidden");
        elements.label.textContent = "ERROR";
        elements.reason.textContent = message;
        elements.analysis.textContent = "Start the FastAPI backend and retrain the model if artefacts are missing.";
    }

    async function getActiveTab() {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        return tab;
    }

    function loadFromGmail() {
        if (isLoadingFromGmail) return;
        isLoadingFromGmail = true;
        getActiveTab().then((tab) => {
            if (!tab || !tab.id) {
                alert("Open a Gmail message before using Get from Gmail.");
                return;
            }
            chrome.tabs.sendMessage(tab.id, { command: "get_email_data" }, (response) => {
                if (chrome.runtime.lastError) {
                    alert("Open a Gmail message before using Get from Gmail.");
                    return;
                }
                if (!response || (!response.subject && !response.body)) {
                    alert("No email content detected in the current tab. Make sure an email is open.");
                    return;
                }

                elements.emailInput.value = [
                    response.sender ? `From: ${response.sender}` : "",
                    response.subject ? `Subject: ${response.subject}` : "",
                    "",
                    response.body || ""
                ].filter((line, i, arr) => !(line === "" && i > 0 && arr[i - 1] === "")).join("\n");
            });
        }).catch(() => {
            alert("Could not access the current tab.");
        }).finally(() => {
            isLoadingFromGmail = false;
        });
    }

    async function analyzeCurrentInput() {
        if (isAnalyzing) return;
        const emailText = elements.emailInput.value.trim();
        if (!emailText) {
            currentPrediction = null;
            currentPayload = null;
            resetResultState();
            alert("Paste email content or load it from Gmail first.");
            return;
        }

        isAnalyzing = true;
        const payload = parseEmail(emailText);
        setLoadingState(true);

        try {
            const result = await runtimeMessage({
                command: "analyze_email",
                payload
            });
            renderResult(result, payload);
            refreshBackendStatus();
            loadHistory();
        } catch (error) {
            renderError(error.message || "Could not connect to the backend.");
            updateServiceStatus("Backend offline", false);
        } finally {
            setLoadingState(false);
            isAnalyzing = false;
        }
    }

    async function submitFeedback(userLabel) {
        if (!currentPrediction || !currentPayload || feedbackInFlight) return;

        feedbackInFlight = true;
        elements.btnFeedbackCorrect.disabled = true;
        elements.btnFeedbackSpam.disabled = true;
        elements.btnFeedbackSafe.disabled = true;

        try {
            const response = await runtimeMessage({
                command: "submit_feedback",
                payload: {
                    prediction_id: currentPrediction.prediction_id,
                    sender: currentPayload.sender,
                    subject: currentPayload.subject,
                    body: currentPayload.body,
                    predicted_label: currentPrediction.label,
                    predicted_confidence: currentPrediction.confidence,
                    user_label: userLabel,
                    source: "extension_popup"
                }
            });
            elements.feedbackStatus.textContent = `Saved feedback (${String(response?.verdict || "ok").replace(/_/g, " ")}).`;
            elements.feedbackStatus.classList.remove("hidden");
            setTimeout(() => {
                elements.feedbackStatus.classList.add("hidden");
            }, 3000);
            loadHistory();
            refreshBackendStatus();
        } catch (error) {
            elements.feedbackStatus.textContent = error.message || "Could not save feedback.";
            elements.feedbackStatus.classList.remove("hidden");
        } finally {
            feedbackInFlight = false;
            elements.btnFeedbackCorrect.disabled = false;
            elements.btnFeedbackSpam.disabled = false;
            elements.btnFeedbackSafe.disabled = false;
        }
    }

    function historyBadgeClass(label) {
        if (label === "Spam") return "history-badge spam";
        if (label === "Not Spam") return "history-badge safe";
        if (label === "whitelisted") return "history-badge whitelisted";
        return "history-badge safe";
    }

    async function loadHistory() {
        const loadId = ++historyLoadCount;
        try {
            const history = await runtimeMessage({ command: "get_scan_history" });
            if (loadId !== historyLoadCount) return;

            const fragment = document.createDocumentFragment();

            if (!Array.isArray(history) || !history.length) {
                const empty = document.createElement("p");
                empty.className = "empty-state";
                empty.textContent = "No scans yet.";
                fragment.appendChild(empty);
            } else {
                history.slice(0, 6).forEach((entry) => {
                    const item = document.createElement("div");
                    item.className = "history-item";

                    const title = document.createElement("div");
                    title.className = "history-title";
                    title.textContent = entry.subject || "(No subject)";

                    const meta = document.createElement("div");
                    meta.className = "history-meta-row";

                    const badgeSpan = document.createElement("span");
                    badgeSpan.className = historyBadgeClass(entry.label);
                    badgeSpan.textContent = entry.label || "Unknown";

                    const confidenceSpan = document.createElement("span");
                    confidenceSpan.textContent = `${Math.round((entry.confidence || 0) * 100)}%`;

                    const dateSpan = document.createElement("span");
                    dateSpan.textContent = formatDate(entry.evaluatedAtUtc);

                    meta.append(badgeSpan, confidenceSpan, dateSpan);

                    const sender = document.createElement("div");
                    sender.className = "history-sender";
                    sender.textContent = entry.sender || entry.senderDomain || "Unknown sender";

                    item.append(title, sender, meta);

                    if (entry.verdict) {
                        const verdict = document.createElement("div");
                        verdict.className = "history-verdict";
                        verdict.textContent = `Feedback: ${String(entry.verdict).replace(/_/g, " ")}`;
                        item.appendChild(verdict);
                    }

                    fragment.appendChild(item);
                });
            }

            elements.historyList.innerHTML = "";
            elements.historyList.appendChild(fragment);
        } catch (error) {
            if (loadId !== historyLoadCount) return;
            elements.historyList.innerHTML = "<p class=\"empty-state\">Could not load scan history.</p>";
        }
    }

    elements.btnGet.addEventListener("click", loadFromGmail);
    elements.btnAnalyze.addEventListener("click", analyzeCurrentInput);
    elements.btnSettings.addEventListener("click", () => {
        chrome.runtime.openOptionsPage().catch(() => {});
    });
    elements.btnClearHistory.addEventListener("click", async () => {
        try {
            await runtimeMessage({ command: "clear_scan_history" });
        } catch (error) {
            elements.serviceStatusText.textContent = "Could not clear history.";
            elements.serviceStatus.classList.add("offline");
        }
        loadHistory();
    });
    elements.btnFeedbackCorrect.addEventListener("click", () => {
        if (!currentPrediction) return;
        submitFeedback(currentPrediction.label === "Spam" ? "Spam" : "Not Spam");
    });
    elements.btnFeedbackSpam.addEventListener("click", () => submitFeedback("Spam"));
    elements.btnFeedbackSafe.addEventListener("click", () => submitFeedback("Not Spam"));

    elements.emailInput.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            event.preventDefault();
            if (!elements.btnAnalyze.disabled) {
                analyzeCurrentInput();
            }
        }
    });

    refreshBackendStatus();
    loadHistory();
});
