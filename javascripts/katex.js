/* KaTeX: render math after instant navigation */
function renderKaTeX() {
  if (typeof renderMathInElement === "undefined") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
    strict: "ignore",
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoredClasses: ["no-math", "md-code__content", "highlight"],
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(renderKaTeX);
}
window.addEventListener("load", renderKaTeX);
