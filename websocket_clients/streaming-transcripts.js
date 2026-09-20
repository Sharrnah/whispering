/* Shared native streaming captions. Full snapshots are authoritative; deltas
 * are metadata, never blindly appended. Works with local file:// overlays. */
(function (root) {
  "use strict";
  class StreamingTranscripts {
    constructor({history = false} = {}) {
      this.revisions = new Map();
      this.pending = new Map();
      this.stable = new Set();
      this.history = history;
      const params = new URLSearchParams(root.location ? root.location.search : "");
      this.source = params.get("source");
      const limit = Number(params.get("live_characters") || 500);
      this.limit = Number.isFinite(limit) ? Math.max(1, Math.min(limit, 10000)) : 500;
    }
    accept(message) {
      if (!message.streaming) return true;
      const source = String(message.audio_source_id || "main");
      if (this.source && source !== this.source) return false;
      const caption = message.type === "streaming_caption";
      const key = JSON.stringify([caption ? "caption" : "transcript", source, message.stream_id]);
      const previous = this.revisions.get(key);
      const revision = Number((caption ? message.display_revision : message.stream_revision) || 0);
      const final = Boolean(caption ? message.display_done : message.final);
      if (previous && (previous.final || revision <= previous.revision)) return false;
      this.revisions.set(key, {revision, final});
      if (this.revisions.size > 128) this.revisions.delete(this.revisions.keys().next().value);
      if (!caption && message.display_mode === "blocks") {
        // Full recognition results remain immediate history, while the backend
        // sends independent live captions. Overlay-only pages omit history.
        return this.history && message.type === "transcript";
      }
      if (final) {
        this.pending.delete(source);
        this.stable.delete(source);
      } else {
        this.pending.set(source, String(message.data || message.text || ""));
        if (caption) this.stable.add(source); else this.stable.delete(source);
      }
      if (caption) {
        // Normalize to the clients' existing live-text rendering path.
        message.type = "processing_data";
        message.final = final;
      }
      return true;
    }
    liveText() {
      return Array.from(this.pending.entries()).map(([source, text]) => {
        if (this.stable.has(source)) return text;
        // Array.from clips code points without splitting emoji surrogate pairs.
        const points = Array.from(text);
        if (points.length <= this.limit) return text;
        let tail = points.slice(-this.limit).join("");
        const space = tail.indexOf(" ");
        if (space > 0) tail = tail.slice(space + 1);
        return "… " + tail.trimStart();
      }).join("\n");
    }
    reset() { this.pending.clear(); this.revisions.clear(); this.stable.clear(); }
  }
  root.StreamingTranscripts = StreamingTranscripts;
  if (typeof module !== "undefined") module.exports = StreamingTranscripts;
})(typeof globalThis !== "undefined" ? globalThis : this);
