(() => {
    const SELECTORS = {
        sender: [
            "span[email]",
            ".gD[email]",
            ".go[email]",
            ".yW span[email]",
            "span[data-hovercard-with-tooltip]"
        ],
        subject: [
            "h2.hP",
            "h1.hP",
            ".hP",
            "[data-thread-perm-id]"
        ],
        body: [
            ".a3s.aiL",
            ".a3s",
            ".ii.gt div[dir]",
            "[role='main'] .a3s",
            ".gs .a3s"
        ]
    };

    function queryFirst(selectors) {
        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    return element;
                }
            } catch (e) {
                // invalid selector, skip
            }
        }
        return null;
    }

    function cleanText(value) {
        return typeof value === "string"
            ? value.replace(/\s+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim()
            : "";
    }

    function getSender() {
        const element = queryFirst(SELECTORS.sender);
        if (!element) return "";
        return cleanText(
            element.getAttribute("email")
            || element.getAttribute("name")
            || element.textContent
            || ""
        );
    }

    function getSubject() {
        const element = queryFirst(SELECTORS.subject);
        if (!element) return "";
        return cleanText(element.textContent || "");
    }

    function getBodyElement() {
        return queryFirst(SELECTORS.body);
    }

    function getBody() {
        const el = getBodyElement();
        if (!el) return "";
        return cleanText(el.innerText || el.textContent || "");
    }

    function getBannerAnchor() {
        const bodyElement = getBodyElement();
        if (!bodyElement) {
            return null;
        }

        return bodyElement.closest(".adn.ads")
            || bodyElement.closest("[data-legacy-message-id]")
            || bodyElement.parentElement
            || bodyElement;
    }

    function getEmailData() {
        return {
            sender: getSender(),
            subject: getSubject(),
            body: getBody()
        };
    }

    window.DomParser = {
        getSender,
        getSubject,
        getBody,
        getBodyElement,
        getBannerAnchor,
        getEmailData
    };
})();
