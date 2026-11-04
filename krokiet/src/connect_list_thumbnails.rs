use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::metadata;
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use czkawka_core::common::image::{ImgResizeOptions, check_if_can_display_image, get_dynamic_image_from_path};
use czkawka_core::re_exported::FirFilterType;
use log::{debug, error};
use slint::{ComponentHandle, Image, Rgba8Pixel, SharedPixelBuffer, Weak};

use crate::{Callabler, GuiState, MainWindow};

pub(crate) const MIN_LIST_THUMBNAIL_SIZE: u32 = 32;
pub(crate) const MAX_LIST_THUMBNAIL_SIZE: u32 = 256;
pub(crate) const DEFAULT_LIST_THUMBNAIL_SIZE: u32 = 64;

const MIN_DECODE_SIDE: u32 = 16;
const MAX_DECODE_SIDE: u32 = 1024;
const CACHED_THUMBNAILS_BYTES_LIMIT: usize = 256 * 1024 * 1024;
// Fast scrolling queues many rows that are no longer visible - the oldest are dropped
const PENDING_REQUESTS_LIMIT: usize = 256;
const MAX_WORKER_THREADS: usize = 4;

thread_local! {
    // Only touched from the Slint event loop thread, because slint::Image is not Send
    static THUMBNAIL_CACHE: RefCell<ThumbnailCache> = RefCell::new(ThumbnailCache::new(CACHED_THUMBNAILS_BYTES_LIMIT));
}

type ThumbnailBuffer = SharedPixelBuffer<Rgba8Pixel>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ThumbnailRequest {
    path: String,
    // Longest side in physical pixels
    side: u32,
}

pub(crate) fn connect_list_thumbnails(app: &MainWindow) {
    let weak = app.as_weak();
    let queue = Arc::new(RequestQueue::default());
    let workers_started = OnceCell::new();

    app.global::<Callabler>().on_get_list_thumbnail(move |path, side, _version| {
        let side = (side.max(0) as u32).clamp(MIN_DECODE_SIDE, MAX_DECODE_SIDE);
        THUMBNAIL_CACHE.with_borrow_mut(|cache| {
            if cache.set_side(side) {
                queue.set_current_side(side);
            }
            if let Some(image) = cache.get(&path) {
                return image;
            }
            if cache.needs_request(&path) && check_if_can_display_image(&path) {
                workers_started.get_or_init(|| spawn_workers(&queue, &weak));
                cache.mark_requested(&path);
                if let Some(dropped) = queue.push(ThumbnailRequest { path: path.to_string(), side }) {
                    cache.forget_request(&dropped);
                }
            }
            Image::default()
        })
    });
}

// Results of a new scan may point at files that changed on disk, or at video thumbnails that did not exist before
pub(crate) fn clear_list_thumbnails() {
    THUMBNAIL_CACHE.with_borrow_mut(ThumbnailCache::clear);
}

pub(crate) fn clamp_list_thumbnail_size(size: f32) -> u32 {
    (size.round().max(0.0) as u32).clamp(MIN_LIST_THUMBNAIL_SIZE, MAX_LIST_THUMBNAIL_SIZE)
}

fn spawn_workers(queue: &Arc<RequestQueue>, weak: &Weak<MainWindow>) {
    let worker_count = thread::available_parallelism().map_or(1, NonZero::get).min(MAX_WORKER_THREADS);
    for worker_idx in 0..worker_count {
        let queue = Arc::clone(queue);
        let weak = weak.clone();
        if let Err(e) = thread::Builder::new()
            .name(format!("list-thumbnail-{worker_idx}"))
            .spawn(move || worker_loop(&queue, &weak))
        {
            error!("Failed to spawn list thumbnail worker: {e}");
        }
    }
}

fn worker_loop(queue: &RequestQueue, weak: &Weak<MainWindow>) {
    loop {
        let request = queue.pop_newest_current();
        let thumbnail = load_thumbnail(&request.path, request.side);
        if weak.upgrade_in_event_loop(move |app| store_thumbnail(&app, request, thumbnail)).is_err() {
            // The event loop is gone, so the app is closing
            return;
        }
    }
}

fn load_thumbnail(path: &str, side: u32) -> Option<ThumbnailBuffer> {
    // Video thumbnail files are left empty when their generation failed
    if metadata(path).map_or(true, |m| m.len() == 0) {
        return None;
    }

    let resize_options = ImgResizeOptions {
        max_width: side,
        max_height: side,
        filter: FirFilterType::Bilinear,
    };
    match get_dynamic_image_from_path(path, Some(resize_options)) {
        Ok(loaded) => {
            let rgba = loaded.image.into_rgba8();
            Some(ThumbnailBuffer::clone_from_slice(rgba.as_raw(), rgba.width(), rgba.height()))
        }
        Err(e) => {
            debug!("Failed to load list thumbnail \"{path}\": {e}");
            None
        }
    }
}

fn store_thumbnail(app: &MainWindow, request: ThumbnailRequest, thumbnail: Option<ThumbnailBuffer>) {
    let stored = THUMBNAIL_CACHE.with_borrow_mut(|cache| cache.finish_request(request, thumbnail.map(Image::from_rgba8)));
    if stored {
        // Visible rows pass this version to get_list_thumbnail, so bumping it makes them query the cache again
        let gui_state = app.global::<GuiState>();
        gui_state.set_list_thumbnails_version(gui_state.get_list_thumbnails_version().wrapping_add(1));
    }
}

#[derive(Default)]
struct RequestQueue {
    requests: Mutex<VecDeque<ThumbnailRequest>>,
    not_empty: Condvar,
    current_side: AtomicU32,
}

impl RequestQueue {
    fn set_current_side(&self, side: u32) {
        self.current_side.store(side, Ordering::Relaxed);
    }

    // Returns the oldest request when it had to be dropped to stay within the limit
    fn push(&self, request: ThumbnailRequest) -> Option<String> {
        let mut requests = self.requests.lock().expect("List thumbnail queue mutex poisoned");
        requests.push_back(request);
        let dropped = if requests.len() > PENDING_REQUESTS_LIMIT { requests.pop_front() } else { None };
        self.not_empty.notify_one();
        dropped.map(|request| request.path)
    }

    // Newest first, because the most recently requested rows are the ones on screen.
    // Requests made before the thumbnail size changed are skipped instead of decoded.
    fn pop_newest_current(&self) -> ThumbnailRequest {
        let mut requests = self.requests.lock().expect("List thumbnail queue mutex poisoned");
        loop {
            while let Some(request) = requests.pop_back() {
                if request.side == self.current_side.load(Ordering::Relaxed) {
                    return request;
                }
            }
            requests = self.not_empty.wait(requests).expect("List thumbnail queue mutex poisoned");
        }
    }
}

struct ThumbnailCache {
    side: u32,
    bytes_limit: usize,
    images: HashMap<String, Image>,
    insertion_order: VecDeque<String>,
    cached_bytes: usize,
    requested: HashSet<String>,
    failed: HashSet<String>,
}

impl ThumbnailCache {
    fn new(bytes_limit: usize) -> Self {
        Self {
            side: 0,
            bytes_limit,
            images: HashMap::new(),
            insertion_order: VecDeque::new(),
            cached_bytes: 0,
            requested: HashSet::new(),
            failed: HashSet::new(),
        }
    }

    // Returns true when the side changed, which drops every image decoded for the old one
    fn set_side(&mut self, side: u32) -> bool {
        if side == self.side {
            return false;
        }
        self.side = side;
        self.images.clear();
        self.insertion_order.clear();
        self.cached_bytes = 0;
        self.requested.clear();
        true
    }

    fn get(&self, path: &str) -> Option<Image> {
        self.images.get(path).cloned()
    }

    fn needs_request(&self, path: &str) -> bool {
        !path.is_empty() && !self.requested.contains(path) && !self.failed.contains(path)
    }

    fn mark_requested(&mut self, path: &str) {
        self.requested.insert(path.to_string());
    }

    fn forget_request(&mut self, path: &str) {
        self.requested.remove(path);
    }

    // Returns true when a new image was stored
    fn finish_request(&mut self, request: ThumbnailRequest, image: Option<Image>) -> bool {
        if request.side != self.side {
            return false;
        }
        let path = request.path;
        self.requested.remove(&path);
        let Some(image) = image else {
            self.failed.insert(path);
            return false;
        };

        let new_bytes = image_bytes(&image);
        if let Some(previous) = self.images.insert(path.clone(), image) {
            self.cached_bytes -= image_bytes(&previous);
        } else {
            self.insertion_order.push_back(path);
        }
        self.cached_bytes += new_bytes;
        self.evict_over_limit();
        true
    }

    fn evict_over_limit(&mut self) {
        // The newest image always stays, even if it alone exceeds the limit
        while self.cached_bytes > self.bytes_limit && self.insertion_order.len() > 1 {
            let Some(oldest) = self.insertion_order.pop_front() else {
                return;
            };
            if let Some(image) = self.images.remove(&oldest) {
                self.cached_bytes -= image_bytes(&image);
            }
        }
    }

    fn clear(&mut self) {
        self.images.clear();
        self.insertion_order.clear();
        self.cached_bytes = 0;
        self.failed.clear();
    }
}

fn image_bytes(image: &Image) -> usize {
    let size = image.size();
    size.width as usize * size.height as usize * size_of::<Rgba8Pixel>()
}

#[cfg(test)]
mod tests {
    use std::{env, fs, process};

    use super::*;

    const SIDE: u32 = 128;

    fn request(path: &str) -> ThumbnailRequest {
        ThumbnailRequest {
            path: path.to_string(),
            side: SIDE,
        }
    }

    fn image_with_side(side: u32) -> Image {
        Image::from_rgba8(ThumbnailBuffer::new(side, side))
    }

    fn cache_with_side() -> ThumbnailCache {
        let mut cache = ThumbnailCache::new(CACHED_THUMBNAILS_BYTES_LIMIT);
        cache.set_side(SIDE);
        cache
    }

    fn finish_ok(cache: &mut ThumbnailCache, path: &str, image_side: u32) -> bool {
        cache.mark_requested(path);
        cache.finish_request(request(path), Some(image_with_side(image_side)))
    }

    #[test]
    fn load_thumbnail_scales_down_keeping_aspect_ratio_and_skips_empty_files() {
        let dir = env::temp_dir();
        let image_path = dir.join(format!("krokiet_list_thumbnail_test_{}.png", process::id()));
        let empty_path = dir.join(format!("krokiet_list_thumbnail_test_{}_empty.png", process::id()));
        image::RgbaImage::new(400, 200).save(&image_path).unwrap();
        fs::write(&empty_path, b"").unwrap();

        let thumbnail = load_thumbnail(&image_path.to_string_lossy(), SIDE);
        let empty = load_thumbnail(&empty_path.to_string_lossy(), SIDE);
        let _ = fs::remove_file(&image_path);
        let _ = fs::remove_file(&empty_path);

        let thumbnail = thumbnail.unwrap();
        assert_eq!((thumbnail.width(), thumbnail.height()), (SIDE, SIDE / 2));
        assert!(empty.is_none());
    }

    #[test]
    fn cache_returns_stored_image_and_stops_requesting_it() {
        let mut cache = cache_with_side();
        assert!(cache.needs_request("a.jpg"));

        cache.mark_requested("a.jpg");
        assert!(!cache.needs_request("a.jpg"));
        assert!(cache.get("a.jpg").is_none());

        assert!(cache.finish_request(request("a.jpg"), Some(image_with_side(4))));
        assert!(cache.get("a.jpg").is_some());
    }

    #[test]
    fn failed_paths_are_not_requested_again_until_cleared() {
        let mut cache = cache_with_side();
        cache.mark_requested("broken.png");
        assert!(!cache.finish_request(request("broken.png"), None));
        assert!(!cache.needs_request("broken.png"));

        cache.clear();
        assert!(cache.needs_request("broken.png"));
    }

    #[test]
    fn empty_path_is_never_requested() {
        assert!(!cache_with_side().needs_request(""));
    }

    #[test]
    fn changing_side_drops_images_and_ignores_results_for_old_side() {
        let mut cache = cache_with_side();
        finish_ok(&mut cache, "a.jpg", 4);
        cache.mark_requested("b.jpg");

        assert!(cache.set_side(SIDE * 2));
        assert!(!cache.set_side(SIDE * 2));
        assert!(cache.get("a.jpg").is_none());
        assert!(cache.needs_request("b.jpg"));
        assert_eq!(cache.cached_bytes, 0);

        assert!(!cache.finish_request(request("b.jpg"), Some(image_with_side(4))));
        assert!(cache.get("b.jpg").is_none());
    }

    #[test]
    fn cache_evicts_oldest_images_over_byte_limit() {
        let one_image_bytes = 8 * 8 * 4;
        let mut cache = ThumbnailCache::new(4 * one_image_bytes);
        cache.set_side(SIDE);
        for idx in 0..4 {
            finish_ok(&mut cache, &format!("{idx}.jpg"), 8);
        }
        assert_eq!(cache.cached_bytes, 4 * one_image_bytes);

        finish_ok(&mut cache, "newest.jpg", 8);

        assert_eq!(cache.cached_bytes, 4 * one_image_bytes);
        assert!(cache.get("0.jpg").is_none());
        assert!(cache.get("1.jpg").is_some());
        assert!(cache.get("newest.jpg").is_some());
    }

    #[test]
    fn storing_same_path_twice_replaces_its_bytes() {
        let mut cache = cache_with_side();
        finish_ok(&mut cache, "a.jpg", 4);
        finish_ok(&mut cache, "a.jpg", 2);
        assert_eq!(cache.insertion_order.len(), 1);
        assert_eq!(cache.cached_bytes, 2 * 2 * 4);
    }

    #[test]
    fn queue_pops_newest_and_drops_oldest_over_limit() {
        let queue = RequestQueue::default();
        queue.set_current_side(SIDE);
        for idx in 0..PENDING_REQUESTS_LIMIT {
            assert_eq!(queue.push(request(&format!("{idx}.jpg"))), None);
        }

        assert_eq!(queue.push(request("newest.jpg")), Some("0.jpg".to_string()));
        assert_eq!(queue.pop_newest_current(), request("newest.jpg"));
        assert_eq!(queue.pop_newest_current(), request(&format!("{}.jpg", PENDING_REQUESTS_LIMIT - 1)));
    }

    #[test]
    fn queue_skips_requests_for_old_side() {
        let queue = RequestQueue::default();
        queue.set_current_side(SIDE);
        queue.push(request("current.jpg"));
        queue.push(ThumbnailRequest {
            path: "stale.jpg".to_string(),
            side: SIDE * 2,
        });

        assert_eq!(queue.pop_newest_current(), request("current.jpg"));
    }

    #[test]
    fn list_thumbnail_size_is_clamped_and_rounded() {
        assert_eq!(clamp_list_thumbnail_size(0.0), MIN_LIST_THUMBNAIL_SIZE);
        assert_eq!(clamp_list_thumbnail_size(-5.0), MIN_LIST_THUMBNAIL_SIZE);
        assert_eq!(clamp_list_thumbnail_size(99.6), 100);
        assert_eq!(clamp_list_thumbnail_size(10_000.0), MAX_LIST_THUMBNAIL_SIZE);
    }
}
