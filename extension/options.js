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
                    requestTimeoutMs: Number(elements.requestTimeout.value) || 30000,
                    historyLimit: Number(elements.historyLimit.value) || 12,
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
        elements.btnCheck.disabled = true;
        try {
            const health = await runtimeMessage({ command: "check_backend_health" });
            const backend = health.feedback_backend || "unknown";
            const model = health.model_version || "unknown";
            setStatus(`Backend online (${backend}). Model: ${model}. /v1/health OK`);
        } catch (error) {
            setStatus(error.message || "Backend is unavailable.");
        } finally {
            elements.btnCheck.disabled = false;
        }
    });

    elements.btnRetrain.addEventListener("click", async () => {
        elements.btnRetrain.disabled = true;
        try {
            setStatus("Retraining model from reviewed feedback. This can take a few minutes...");
            const result = await runtimeMessage({ command: "retrain_model" });
            const version = result.model_version || "unknown";
            const backend = result.feedback_backend || "unknown";
            const used = result.feedback_rows_used ?? 0;
            setStatus(`Retrained ${version} via ${backend}. Feedback used: ${used}.`);
        } catch (error) {
            setStatus(error.message || "Could not retrain model.");
        } finally {
            elements.btnRetrain.disabled = false;
        }
    });

    loadSettings().catch((error) => setStatus(error.message || "Could not load settings."));
});
