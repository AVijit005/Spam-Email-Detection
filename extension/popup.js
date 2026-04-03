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

    let currentPayload = null;
    let currentPrediction = null;

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

    function resetResultState() {
        elements.resultBox.classList.remove("hidden", "visible", "spam", "safe", "whitelisted", "error");
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
        resetResultState();
        elements.resultBox.classList.add("visible");
        if (isLoading) {
            elements.loaderContainer.classList.remove("hidden");
        } else {
            elements.loaderContainer.classList.add("hidden");
            elements.resultContent.classList.remove("hidden");
        }
    }

    function updateServiceStatus(text, isOnline) {
        elements.serviceStatus.classList.toggle("offline", !isOnline);
        elements.serviceStatusText.textContent = text;
    }

    function formatDate(isoString) {
        if (!isoString) {
            return "";
        }

        const date = new Date(isoString);
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
            const version = health.model_version && health.model_version !== "untrained"
                ? ` • ${health.model_version}`
                : "";
            updateServiceStatus(`Backend online${version}`, Boolean(health.model_loaded));

            const metaBits = [
                health.trained_at_utc ? `Trained ${formatDate(health.trained_at_utc)}` : null,
                `Feedback ${health.feedback_count}`,
                `Whitelist ${health.user_whitelist_count}`
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
        const normalized = text.replace(/\r\n/g, "\n");
        const senderMatch = normalized.match(/^\s*From:\s*(.+)$/im);
        const subjectMatch = normalized.match(/^\s*Subject:\s*(.+)$/im);

        let body = normalized;
        if (senderMatch || subjectMatch) {
            body = normalized
                .replace(/^\s*From:\s*.+$/im, "")
                .replace(/^\s*Subject:\s*.+$/im, "")
                .trim();
        }

        return {
            sender: senderMatch ? senderMatch[1].trim() : "",
            subject: subjectMatch ? subjectMatch[1].trim() : "",
            body: body.trim()
        };
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
            item.textContent = entry;
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
    }

    function renderResult(data, payload) {
        currentPrediction = data;
        currentPayload = payload;

        resetResultState();
        elements.resultBox.classList.add("visible");
        elements.resultContent.classList.remove("hidden");

        const cssClass = data.label === "Spam"
            ? "spam"
            : data.label === "whitelisted"
                ? "whitelisted"
                : "safe";

        const displayLabel = data.label === "whitelisted"
            ? "WHITELISTED"
            : data.label.toUpperCase();

        elements.resultBox.classList.add(cssClass);
        elements.label.textContent = displayLabel;
        elements.confidence.textContent = `Confidence: ${Math.round((data.confidence || 0) * 100)}%`;
        elements.reason.textContent = data.reason || "";
        elements.analysis.textContent = data.analysis || "";

        setTimeout(() => {
            elements.confidenceFill.style.width = `${Math.round((data.confidence || 0) * 100)}%`;
        }, 80);

        if (data.rule_layer) {
            addMetaChip(`Layer: ${data.rule_layer}`);
        }
        if (data.model_version) {
            addMetaChip(data.model_version);
        }
        if (data.sender_domain) {
            addMetaChip(data.sender_domain);
        }
        if (data.evaluated_at_utc) {
            addMetaChip(formatDate(data.evaluated_at_utc));
        }
        if (data.signals?.length) {
            data.signals.slice(0, 3).forEach(addMetaChip);
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

    async function loadFromGmail() {
        const tab = await getActiveTab();
        chrome.tabs.sendMessage(tab.id, { command: "get_email_data" }, (response) => {
            if (chrome.runtime.lastError) {
                alert("Open a Gmail message before using Get from Gmail.");
                return;
            }
            if (!response || (!response.subject && !response.body)) {
                alert("No email content detected in the current tab.");
                return;
            }

            elements.emailInput.value = [
                `From: ${response.sender || ""}`,
                `Subject: ${response.subject || ""}`,
                "",
                response.body || ""
            ].join("\n");
        });
    }

    async function analyzeCurrentInput() {
        const emailText = elements.emailInput.value.trim();
        if (!emailText) {
            alert("Paste email content or load it from Gmail first.");
            return;
        }

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
        }
    }

    async function submitFeedback(userLabel) {
        if (!currentPrediction || !currentPayload) {
            return;
        }

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
            elements.feedbackStatus.textContent = `Saved feedback (${response.verdict.replace("_", " ")}).`;
            elements.feedbackStatus.classList.remove("hidden");
            loadHistory();
            refreshBackendStatus();
        } catch (error) {
            elements.feedbackStatus.textContent = error.message || "Could not save feedback.";
            elements.feedbackStatus.classList.remove("hidden");
        }
    }

    function historyBadgeClass(label) {
        return label === "Spam" ? "history-badge spam" : "history-badge safe";
    }

    async function loadHistory() {
        try {
            const history = await runtimeMessage({ command: "get_scan_history" });
            elements.historyList.innerHTML = "";

            if (!history.length) {
                elements.historyList.innerHTML = "<p class=\"empty-state\">No scans yet.</p>";
                return;
            }

            history.slice(0, 6).forEach((entry) => {
                const item = document.createElement("div");
                item.className = "history-item";

                const title = document.createElement("div");
                title.className = "history-title";
                title.textContent = entry.subject || "(No subject)";

                const meta = document.createElement("div");
                meta.className = "history-meta-row";
                meta.innerHTML = `
                    <span class="${historyBadgeClass(entry.label)}">${entry.label}</span>
                    <span>${Math.round((entry.confidence || 0) * 100)}%</span>
                    <span>${formatDate(entry.evaluatedAtUtc)}</span>
                `;

                const sender = document.createElement("div");
                sender.className = "history-sender";
                sender.textContent = entry.sender || entry.senderDomain || "Unknown sender";

                item.append(title, sender, meta);

                if (entry.verdict) {
                    const verdict = document.createElement("div");
                    verdict.className = "history-verdict";
                    verdict.textContent = `Feedback: ${entry.verdict.replace("_", " ")}`;
                    item.appendChild(verdict);
                }

                elements.historyList.appendChild(item);
            });
        } catch (error) {
            elements.historyList.innerHTML = "<p class=\"empty-state\">Could not load scan history.</p>";
        }
    }

    elements.btnGet.addEventListener("click", loadFromGmail);
    elements.btnAnalyze.addEventListener("click", analyzeCurrentInput);
    elements.btnSettings.addEventListener("click", () => chrome.runtime.openOptionsPage());
    elements.btnClearHistory.addEventListener("click", async () => {
        await runtimeMessage({ command: "clear_scan_history" });
        loadHistory();
    });
    elements.btnFeedbackCorrect.addEventListener("click", () => {
        if (!currentPrediction) {
            return;
        }
        submitFeedback(currentPrediction.label === "Spam" ? "Spam" : "Not Spam");
    });
    elements.btnFeedbackSpam.addEventListener("click", () => submitFeedback("Spam"));
    elements.btnFeedbackSafe.addEventListener("click", () => submitFeedback("Not Spam"));

    elements.emailInput.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            analyzeCurrentInput();
        }
    });

    refreshBackendStatus();
    loadHistory();
});
