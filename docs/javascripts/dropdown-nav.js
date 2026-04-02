/**
 * Dropdown Mega Menu for MkDocs Material
 * Groups flat tabs into dropdown menus on hover.
 * Uses position:fixed to avoid overflow clipping.
 */
(function () {
  "use strict";

  function getBaseUrl() {
    var path = window.location.pathname;
    var idx = path.indexOf("/credit-scoring-guidebook/");
    if (idx >= 0) return path.substring(0, idx + "/credit-scoring-guidebook/".length);
    return "/";
  }

  var GROUPS = [
    {
      label: "스코어카드",
      sublabel: "Traditional Scorecard",
      indexPath: "scorecard/",
      tabMatches: ["Traditional Scorecard", "1. 개요", "2. 이론", "3. 변수 선정", "4. 모델링", "5. 스코어카드", "부록"],
      items: [
        { text: "1. 개요", path: "scorecard/part1_overview/" },
        { text: "2. 이론", path: "scorecard/part2_theory/" },
        { text: "3. 변수 선정", path: "scorecard/part3_variable_selection/" },
        { text: "4. 모델링", path: "scorecard/part4_modeling/" },
        { text: "5. 스코어카드", path: "scorecard/part5_scorecard/" },
        { text: "부록", path: "scorecard/appendix/" }
      ]
    },
    {
      label: "머신러닝",
      sublabel: "Machine Learning",
      indexPath: "ml/",
      tabMatches: ["머신러닝", "1. 개요", "2. 뉴럴넷", "3. 트리 앙상블", "4. 평가와 해석", "5. 모델 검증", "부록"],
      items: [
        { text: "1. 개요", path: "ml/part1_overview/" },
        { text: "2. 뉴럴넷", path: "ml/part2_neural_net/" },
        { text: "3. 트리 앙상블", path: "ml/part3_tree_ensemble/" },
        { text: "4. 해석과 설명", path: "ml/part4_evaluation/" },
        { text: "5. 모델 검증", path: "ml/part5_validation/" },
        { text: "부록", path: "ml/appendix/" }
      ]
    }
  ];

  function textOf(el) {
    var text = "";
    el.childNodes.forEach(function(n) {
      if (n.nodeType === 3) text += n.textContent;
    });
    return text.trim() || (el.textContent || "").trim();
  }

  var activePanel = null;
  var hideTimeout = null;

  function showPanel(panel, triggerEl) {
    if (hideTimeout) { clearTimeout(hideTimeout); hideTimeout = null; }
    if (activePanel && activePanel !== panel) hidePanel(activePanel);
    var rect = triggerEl.getBoundingClientRect();
    panel.style.top = rect.bottom + "px";
    panel.style.left = (rect.left + rect.width / 2) + "px";
    panel.classList.add("md-dropdown-panel--visible");
    activePanel = panel;
  }

  function hidePanel(panel) {
    if (!panel) return;
    panel.classList.remove("md-dropdown-panel--visible");
    if (activePanel === panel) activePanel = null;
  }

  function scheduleHide(panel) {
    hideTimeout = setTimeout(function () { hidePanel(panel); }, 120);
  }

  function cancelHide() {
    if (hideTimeout) { clearTimeout(hideTimeout); hideTimeout = null; }
  }

  function init() {
    var tabList = document.querySelector(".md-tabs__list");
    if (!tabList) return;
    if (tabList.querySelector(".md-tabs__item--dropdown")) return;

    var baseUrl = getBaseUrl();
    var currentPath = window.location.pathname;
    var items = Array.from(tabList.querySelectorAll(".md-tabs__item"));

    // Collect all grouped tab texts
    var groupedTexts = {};
    GROUPS.forEach(function (g) {
      g.tabMatches.forEach(function (t) { groupedTexts[t] = true; });
    });

    var fragment = document.createDocumentFragment();

    // Standalone tabs (Preface etc.)
    items.forEach(function (item) {
      var link = item.querySelector(".md-tabs__link");
      if (link && !groupedTexts[textOf(link)]) {
        fragment.appendChild(item);
      }
    });

    // Remove old panels
    document.querySelectorAll(".md-dropdown-panel").forEach(function (el) { el.remove(); });

    // Create dropdown groups
    GROUPS.forEach(function (group) {
      var wrapper = document.createElement("li");
      wrapper.className = "md-tabs__item md-tabs__item--dropdown";

      // Index page href
      var firstHref = baseUrl + (group.indexPath || "");

      var trigger = document.createElement("a");
      trigger.className = "md-tabs__link md-tabs__link--dropdown";
      trigger.href = firstHref;
      trigger.textContent = group.label;
      if (group.sublabel) trigger.setAttribute("data-sublabel", group.sublabel);
      wrapper.appendChild(trigger);

      // Active check
      var hasActive = false;
      group.items.forEach(function (mi) {
        if (mi.path && currentPath.indexOf(mi.path) >= 0) hasActive = true;
      });
      if (hasActive) wrapper.classList.add("md-tabs__item--active");

      // Panel (appended to body)
      var panel = document.createElement("div");
      panel.className = "md-dropdown-panel";

      group.items.forEach(function (mi) {
        if (mi.divider) {
          var div = document.createElement("div");
          div.className = "md-dropdown-panel__divider";
          div.textContent = mi.text;
          panel.appendChild(div);
          return;
        }
        var entry = document.createElement("a");
        entry.className = "md-dropdown-panel__link";
        entry.href = baseUrl + mi.path;
        entry.textContent = mi.text;
        if (currentPath.indexOf(mi.path) >= 0) {
          entry.classList.add("md-dropdown-panel__link--active");
        }
        panel.appendChild(entry);
      });

      document.body.appendChild(panel);

      // Hover events
      wrapper.addEventListener("mouseenter", function () { showPanel(panel, wrapper); });
      wrapper.addEventListener("mouseleave", function () { scheduleHide(panel); });
      panel.addEventListener("mouseenter", function () { cancelHide(); });
      panel.addEventListener("mouseleave", function () { scheduleHide(panel); });

      // Click navigates to index page (href is already set)

      fragment.appendChild(wrapper);
    });

    tabList.innerHTML = "";
    tabList.appendChild(fragment);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  document.addEventListener("DOMContentSwitch", function () {
    document.querySelectorAll(".md-dropdown-panel").forEach(function (el) { el.remove(); });
    var tabList = document.querySelector(".md-tabs__list");
    if (tabList) {
      tabList.querySelectorAll(".md-tabs__item--dropdown").forEach(function (el) {
        el.classList.remove("md-tabs__item--dropdown");
      });
    }
    init();
  });
})();
