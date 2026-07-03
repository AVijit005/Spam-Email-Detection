(() => {
    const SELECTORS = {
        sender: [".gD[email]", ".go [email]"],
        subject: ["h2.hP", ".hP"],
        body: [".a3s.aiL", ".a3s"]
    };

    function queryFirst(selectors) {
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                return element;
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
        return cleanText(element?.getAttribute("email") || element?.textContent || "");
    }

    function getSubject() {
        const element = queryFirst(SELECTORS.subject);
        return cleanText(element?.textContent || "");
    }

    function getBodyElement() {
        return queryFirst(SELECTORS.body);
    }

    function getBody() {
        return cleanText(getBodyElement()?.innerText || "");
    }

    function getBannerAnchor() {
        const bodyElement = getBodyElement();
        if (!bodyElement) {
            return null;
        }

        return bodyElement.closest(".adn.ads")
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
