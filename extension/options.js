document.addEventListener("DOMContentLoaded", async () => {
    const elements = {
        form: document.getElementById("settings-form"),
        apiBaseUrl: document.getElementById("api-base-url"),
        requestTimeout: document.getElementById("request-timeout"),
        historyLimit: document.getElementById("history-limit"),
        autoScanEnabled: document.getElementById("auto-scan-enabled"),
        btnCheck: document.getElementById("btn-check"),
        btnRetrain: document.getElementById("btn-retrain"),
        status: document.getElementById("status")
    };

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

    function setStatus(text) {
        elements.status.textContent = text;
    }

    async function loadSettings() {
        const settings = await runtimeMessage({ command: "get_settings" });
        elements.apiBaseUrl.value = settings.apiBaseUrl;
        elements.requestTimeout.value = settings.requestTimeoutMs;
        elements.historyLimit.value = settings.historyLimit;
        elements.autoScanEnabled.checked = Boolean(settings.autoScanEnabled);
    }

    elements.form.addEventListener("submit", async (event) => {
        event.preventDefault();
        try {
            const settings = await runtimeMessage({
                command: "save_settings",
                payload: {
                    apiBaseUrl: elements.apiBaseUrl.value,
                    requestTimeoutMs: Number(elements.requestTimeout.value),
                    historyLimit: Number(elements.historyLimit.value),
                    autoScanEnabled: elements.autoScanEnabled.checked
                }
            });
            await runtimeMessage({ command: "clear_prediction_cache" });
            setStatus(`Saved. Backend URL: ${settings.apiBaseUrl}`);
        } catch (error) {
            setStatus(error.message || "Could not save settings.");
        }
    });

    elements.btnCheck.addEventListener("click", async () => {
        try {
            const health = await runtimeMessage({ command: "check_backend_health" });
            setStatus(`Backend online (${health.feedback_backend}). Model: ${health.model_version}. /v1/health OK`);
        } catch (error) {
            setStatus(error.message || "Backend is unavailable.");
        }
    });

    elements.btnRetrain.addEventListener("click", async () => {
        try {
            setStatus("Retraining model from reviewed feedback. This can take a few minutes...");
            const result = await runtimeMessage({ command: "retrain_model" });
            setStatus(
                `Retrained ${result.model_version} via ${result.feedback_backend}. Feedback used: ${result.feedback_rows_used}.`
            );
        } catch (error) {
            setStatus(error.message || "Could not retrain model.");
        }
    });

    loadSettings().catch((error) => setStatus(error.message || "Could not load settings."));
});
