"use strict";
// Content stays visible when JavaScript or an animation dependency is unavailable.
if (window.gsap && window.ScrollTrigger) {
  gsap.registerPlugin(ScrollTrigger);
  const releaseMotion = gsap.matchMedia();
  releaseMotion.add("(prefers-reduced-motion: no-preference)", () => {
    gsap.utils.toArray(".release-card").forEach((card, index) => {
      gsap.from(card, {
        y: 24 + index * 14,
        duration: .8,
        ease: "power2.out",
        clearProps: "transform",
        scrollTrigger: {trigger: card, start: "top 94%", once: true},
      });
      gsap.fromTo(card.querySelector(".release-image img"),
        {scale: .8, opacity: .65},
        {scale: 1, opacity: 1, ease: "none", scrollTrigger: {
          trigger: card, start: "top 95%", end: "top 55%", scrub: .5,
        }},
      );
    });
  });
}
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const videos = document.querySelectorAll("video");
videos.forEach(video => {
  video.addEventListener("error", () => {
    if (video.parentElement.querySelector(".media-error")) return;
    const message = document.createElement("p");
    message.className = "media-error";
    message.textContent = "The video could not load. ";
    const link = document.createElement("a");
    link.href = video.querySelector("source").src;
    link.textContent = "Open the MP4 directly";
    message.append(link);
    video.after(message);
  });
  video.querySelector("source")?.addEventListener("error", () => video.dispatchEvent(new Event("error")));
});
// Only the lead film starts automatically, once, when visible. Native controls
// remain available; scrolling never overrides a visitor's subsequent pause.
const lead = document.querySelector("[data-autoplay]");
if (lead && "IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) return;
    observer.disconnect();
    if (!reducedMotion.matches && !navigator.connection?.saveData) lead.play().catch(() => {});
  }, {threshold: .35});
  observer.observe(lead);
}
reducedMotion.addEventListener("change", e => {if (e.matches) videos.forEach(video => video.pause());});
document.addEventListener("visibilitychange", () => {if (document.hidden) videos.forEach(video => video.pause());});
const copy = document.querySelector("#copy");
if (navigator.clipboard?.writeText) {
  copy.hidden = false;
  copy.addEventListener("click", async () => {
    const status = document.querySelector("#copy-status");
    try {
      await navigator.clipboard.writeText(document.querySelector("#commands").textContent);
      status.textContent = "Commands copied to clipboard.";
    } catch {
      status.textContent = "Clipboard access is unavailable. Select the commands above to copy them.";
    }
  });
}
