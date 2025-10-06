; (() => {
    const screen = document.getElementById("screen")
    const queryInput = document.getElementById("queryInput")
    const suggestBtn = document.getElementById("suggestBtn")
    const runBtn = document.getElementById("runBtn")
    const suggestionsEl = document.getElementById("suggestions")
    const tmpl = document.getElementById("suggestionItemTmpl")

    let suggestions = []
    let activeIndex = -1 // which suggestion is highlighted

    function printLine({ text, isError = false, isCmd = false, prefix = "PS C:\\Users\\You>" }) {
        const line = document.createElement("div")
        line.className = "line" + (isError ? " error" : "")
        if (isCmd) {
            line.innerHTML = `<span class="prefix">${prefix} </span><span class="cmd">${escapeHtml(text)}</span>`
        } else {
            line.textContent = text
        }
        screen.appendChild(line)
        screen.scrollTop = screen.scrollHeight
    }

    function escapeHtml(s) {
        return s.replace(
            /[&<>"']/g,
            (c) =>
                ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;",
                })[c],
        )
    }

    async function fetchSuggestions(query) {
        if (!query.trim()) {
            clearSuggestions()
            return
        }
        setLoading(true)
        try {
            const res = await fetch("/suggest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            })
            if (!res.ok) throw new Error(`Suggest failed: ${res.status}`)
            const data = await res.json()
            suggestions = Array.isArray(data) ? data : []
            renderSuggestions()
        } catch (err) {
            printLine({ text: `[suggest] ${err.message}`, isError: true })
        } finally {
            setLoading(false)
        }
    }

    async function runCommand(command) {
        if (!command.trim()) return
        printLine({ text: command, isCmd: true })

        setLoading(true)
        try {
            const res = await fetch("/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ command }),
            })
            const data = await res.json()
            if (!res.ok) {
                const msg = data?.error || `HTTP ${res.status}`
                printLine({ text: msg, isError: true })
                return
            }
            const stdout = data?.stdout || ""
            const stderr = data?.stderr || ""
            if (stdout) stdout.split("\n").forEach((line) => printLine({ text: line }))
            if (stderr) stderr.split("\n").forEach((line) => printLine({ text: line, isError: true }))
        } catch (err) {
            printLine({ text: `[run] ${err.message}`, isError: true })
        } finally {
            setLoading(false)
        }
    }

    function renderSuggestions() {
        clearSuggestions(false)
        suggestions.forEach((s, idx) => {
            const node = tmpl.content.firstElementChild.cloneNode(true)
            node.querySelector(".suggestion-cmd").textContent = s.command || ""
            node.querySelector(".suggestion-desc").textContent = s.description || ""
            node.querySelector(".suggestion-meta").textContent =
                typeof s.score === "number" ? `score: ${s.score.toFixed(3)}` : ""
            if (idx === activeIndex) node.classList.add("active")
            node.addEventListener("click", () => {
                setActive(idx)
                acceptActive() // click selects and runs
            })
            node.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault()
                    setActive(idx)
                    acceptActive()
                }
            })
            suggestionsEl.appendChild(node)
        })
    }

    function clearSuggestions(resetIndex = true) {
        suggestionsEl.innerHTML = ""
        if (resetIndex) activeIndex = -1
    }

    function setActive(next) {
        activeIndex = next
        renderSuggestions()
    }

    function moveActive(delta) {
        if (!suggestions.length) return
        if (activeIndex === -1 && delta > 0) {
            activeIndex = 0
        } else {
            activeIndex = (activeIndex + delta + suggestions.length) % suggestions.length
        }
        renderSuggestions()
    }

    function acceptActive() {
        if (activeIndex < 0 || activeIndex >= suggestions.length) return
        const cmd = suggestions[activeIndex].command || ""
        queryInput.value = cmd // surface the chosen command
        runCommand(cmd)
        suggestions = []
        clearSuggestions()
        queryInput.focus()
    }

    function setLoading(loading) {
        if (loading) {
            runBtn.disabled = true
            suggestBtn.disabled = true
            runBtn.dataset.loading = "true"
            suggestBtn.dataset.loading = "true"
        } else {
            runBtn.disabled = false
            suggestBtn.disabled = false
            delete runBtn.dataset.loading
            delete suggestBtn.dataset.loading
        }
    }

    // UI adornments for loading
    const origRunText = "Run"
    const origSuggestText = "Suggest"
    const obs = new MutationObserver(() => {
        runBtn.textContent = runBtn.dataset.loading ? "Running…" : origRunText
        suggestBtn.textContent = suggestBtn.dataset.loading ? "Suggesting…" : origSuggestText
    })
    obs.observe(runBtn, { attributes: true, attributeFilter: ["data-loading"] })
    obs.observe(suggestBtn, { attributes: true, attributeFilter: ["data-loading"] })

    // Events
    suggestBtn.addEventListener("click", () => fetchSuggestions(queryInput.value))
    runBtn.addEventListener("click", () => {
        const cmd = activeIndex >= 0 ? suggestions[activeIndex]?.command || "" : queryInput.value
        runCommand(cmd)
    })

    queryInput.addEventListener("keydown", (e) => {
        // Keyboard shortcuts
        if (e.key === "ArrowDown") {
            e.preventDefault()
            moveActive(1)
        } else if (e.key === "ArrowUp") {
            e.preventDefault()
            moveActive(-1)
        } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            // Ctrl/Cmd+Enter => run top suggestion
            e.preventDefault()
            if (suggestions.length > 0) {
                setActive(0)
                acceptActive()
            } else runCommand(queryInput.value)
        } else if (e.key === "Tab") {
            // Tab => accept active suggestion into input
            if (activeIndex >= 0 && suggestions[activeIndex]) {
                e.preventDefault()
                queryInput.value = suggestions[activeIndex].command || ""
            }
        } else if (e.key === "Enter") {
            // Enter => get suggestions (first), press again to run
            e.preventDefault()
            if (!suggestions.length) {
                fetchSuggestions(queryInput.value)
            } else {
                acceptActive()
            }
        }
    })

    // Initial greeting
    printLine({
        text: "Windows Terminal-like UI ready. Type a request and press Enter to get suggestions,",
        isError: false,
    })
    printLine({ text: "then use Arrow keys to choose, Tab to autocomplete, and Ctrl+Enter to run.", isError: false })
})()
