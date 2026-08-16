document.addEventListener("DOMContentLoaded", function () {
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;

  btn.addEventListener("click", function () {
    var root = document.documentElement;
    var current = root.getAttribute("data-theme");
    var systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var effective = current || (systemDark ? "dark" : "light");
    var next = effective === "dark" ? "light" : "dark";

    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      /* ignore (private browsing, etc.) */
    }
  });
});
