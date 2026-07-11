(() => {
    const SELECTORS = {
        sender: [
            ".gD[email]",
            ".gD",
            "[email]",
            ".go [email]",
            "table[role='presentation'] .gD[email]",
            "span[email]"
        ],
        subject: [
            "h2.hP",
            ".hP",
            "[data-thread-perm-id] h2",
            "div[role='heading'][tabindex='-1']",
            "[data-thread-perm-id] .hP",
            ".hP.h1"
        ],
        body: [
            ".a3s.aiL",
            ".a3s",
            "div[role='document'] .ii",
            "div[role='listitem'] .ii.gt",
            ".ii.gt",
            "[role='document']",
            ".ii",
            "[role='article']"
        ]
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
        const candidate = queryFirst(SELECTORS.sender);
        if (!candidate) {
            return "";
        }
        const emailAttr = candidate.getAttribute("email");
        if (emailAttr) {
            return cleanText(emailAttr);
        }
        const text = cleanText(candidate.textContent || "");
        const match = text.match(/<([^>]+@[^>]+)>/);
        if (match) {
            return match[1];
        }
        const bare = text.match(/([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})/i);
        return bare ? bare[1] : text;
    }

    function getSubject() {
        const element = queryFirst(SELECTORS.subject);
        return cleanText(element?.textContent || "");
    }

    function getBodyElement() {
        const specific = queryFirst(SELECTORS.body);
        if (specific) {
            return specific;
        }
        return heuristicBody();
    }

    function heuristicBody() {
        const root = document.querySelector("[role='main']") || document.body;
        const candidates = Array.from(root.querySelectorAll("div"))
            .filter((el) => {
                const t = (el.innerText || "").trim();
                return t.length > 80 && t.length < 20000;
            })
            .sort((a, b) => (b.innerText || "").length - (a.innerText || "").length);
        return candidates[0] || null;
    }

    function getBody() {
        const element = getBodyElement();
        if (!element) {
            return "";
        }
        const clone = element.cloneNode(true);
        clone.querySelectorAll("style, script, noscript, img, table[role='presentation']").forEach((node) => node.remove());
        return cleanText(clone.innerText || element.innerText || "");
    }

    function getBannerAnchor() {
        const bodyElement = getBodyElement();
        if (!bodyElement) {
            return document.querySelector("[role='main']") || document.querySelector(".nH") || null;
        }

        return bodyElement.closest(".nH")
            || bodyElement.closest("[role='main']")
            || bodyElement.closest("[role='listitem']")
            || bodyElement.parentElement
            || bodyElement;
    }

    function isMessageOpen() {
        return !!document.querySelector(".hP, h2.hP, [role='document'], .a3s, .a3s.aiL");
    }

    function getEmailData() {
        if (!isMessageOpen()) {
            return { sender: "", subject: "", body: "" };
        }
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
