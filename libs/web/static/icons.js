(() => {
  const svgNS = "http://www.w3.org/2000/svg";
  const icons = {
    "badge-dollar-sign": '<path d="M12 3l2.5 2.1 3.2-.2.9 3 2.5 2.1-1.3 2.9 1.3 2.9-2.5 2.1-.9 3-3.2-.2L12 21l-2.5-2.1-3.2.2-.9-3L3 14l1.3-2.9L3 8.2l2.5-2.1.9-3 3.2.2L12 3z"/><path d="M12 7v10"/><path d="M15 9.5c-.6-.9-1.7-1.3-3-1.3-1.7 0-3 1-3 2.3 0 3.4 6 1.3 6 4.6 0 1.3-1.3 2.4-3 2.4-1.4 0-2.5-.5-3.2-1.4"/>',
    "bar-chart-3": '<path d="M3 3v18h18"/><path d="M7 16V9"/><path d="M12 16V5"/><path d="M17 16v-4"/>',
    brain: '<path d="M8.5 14.5A3.5 3.5 0 0 1 5 11c0-1.6 1.1-3 2.6-3.4A3.6 3.6 0 0 1 14 5.3 3.7 3.7 0 0 1 19 8.8a3.7 3.7 0 0 1-.9 7.2H16"/><path d="M12 5v14"/><path d="M8 19a3 3 0 0 1-3-3v-1"/><path d="M16 19a3 3 0 0 0 3-3v-1"/><path d="M8.5 10.5h2"/><path d="M13.5 10.5h2"/><path d="M9 15h2"/><path d="M13 15h2"/>',
    "calendar-search": '<path d="M8 2v4"/><path d="M16 2v4"/><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 10h18"/><circle cx="11" cy="15" r="2.5"/><path d="M13 17l2 2"/>',
    copy: '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "message-square-text": '<path d="M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 8h8"/><path d="M8 12h6"/>',
    play: '<polygon points="6,4 20,12 6,20"/>',
    "refresh-cw": '<path d="M21 12a9 9 0 0 1-15.4 6.4L3 16"/><path d="M3 21v-5h5"/><path d="M3 12A9 9 0 0 1 18.4 5.6L21 8"/><path d="M21 3v5h-5"/>',
    "sliders-horizontal": '<path d="M3 6h10"/><path d="M17 6h4"/><path d="M3 12h4"/><path d="M11 12h10"/><path d="M3 18h12"/><path d="M19 18h2"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    sparkles: '<path d="M12 3l1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4z"/><path d="M5 15l.8 2.2L8 18l-2.2.8L5 21l-.8-2.2L2 18l2.2-.8z"/><path d="M19 13l.7 1.8 1.8.7-1.8.7L19 18l-.7-1.8-1.8-.7 1.8-.7z"/>',
    swords: '<path d="M14.5 17.5L3 6V3h3l11.5 11.5"/><path d="M13 19l6-6"/><path d="M16 16l3 3"/><path d="M19 21l2-2"/><path d="M9.5 17.5L21 6V3h-3L6.5 14.5"/><path d="M11 19l-6-6"/><path d="M8 16l-3 3"/><path d="M5 21l-2-2"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9"/><path d="M16 3.1a4 4 0 0 1 0 7.8"/>',
  };

  function createIcon(name) {
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("dashboard-icon", `icon-${name}`);
    svg.innerHTML = icons[name] || '<circle cx="12" cy="12" r="9"/>';
    return svg;
  }

  function createIcons() {
    document.querySelectorAll("[data-lucide]").forEach((node) => {
      const name = node.getAttribute("data-lucide") || "circle";
      node.replaceChildren(createIcon(name));
    });
  }

  window.lucide = { createIcons };
})();
