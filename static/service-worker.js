const CACHE_NAME = "msl-translator-v1";

const ASSETS = [
    "/",
    "/static/style.css",
    "/static/script.js",
    "/static/manifest.json"
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener("fetch", (event) => {
    if (
        event.request.url.includes("/video_feed") ||
        event.request.url.includes("/prediction")
    ) {
        return;
    }

    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            return cachedResponse || fetch(event.request);
        })
    );
});
