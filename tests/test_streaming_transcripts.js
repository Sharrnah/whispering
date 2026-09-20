"use strict";
const assert = require("node:assert/strict");
const StreamingTranscripts = require("../websocket_clients/streaming-transcripts.js");
const captions = new StreamingTranscripts();
const event = (text, revision, final = false, source = "main") => ({
  type: final ? "transcript" : "processing_data", data: text, text,
  streaming: true, stream_id: source + "-session", stream_revision: revision,
  final, audio_source_id: source,
});
assert(captions.accept(event("hello", 1)));
assert(captions.accept(event("hello world", 2)));
assert.equal(captions.liveText(), "hello world");
assert(!captions.accept(event("stale", 1)));
assert(captions.accept(event("game", 1, false, "game")));
assert(captions.accept(event("hello world", 3, true)));
assert.equal(captions.liveText(), "game");
assert(!captions.accept(event("duplicate", 3, true)));
assert(!captions.accept(event("late", 4)));
captions.reset();
assert.equal(captions.liveText(), "");
assert(captions.accept(event("restart", 1)));
captions.limit = 8;
captions.accept(event("old words 😀newest", 2));
assert.equal(captions.liveText(), "… 😀newest");
captions.source = "game";
assert(!captions.accept(event("filtered", 3)));
console.log("Streaming overlay ordering, source isolation, reconnect and Unicode checks passed.");

const overlay = new StreamingTranscripts();
const history = new StreamingTranscripts({history: true});
const final = {...event("Full transcript for history", 1, true), display_mode: "blocks"};
assert(!overlay.accept({...final}));
assert(history.accept({...final}));
const caption = (text, revision, done = false) => ({type: "streaming_caption", data: text,
  streaming: true, display_mode: "blocks", stream_id: "main-session", audio_source_id: "main",
  display_revision: revision, display_done: done});
const first = caption("A stable first line\nwith a second line", 1);
assert(overlay.accept(first));
assert.equal(first.type, "processing_data");
overlay.limit = 4; // Stable lines must never pass through rolling-tail clipping.
assert.equal(overlay.liveText(), "A stable first line\nwith a second line");
assert(!overlay.accept(caption("old", 1)));
assert(overlay.accept(caption("Next complete phrase", 2)));
assert.equal(overlay.liveText(), "Next complete phrase");
assert(overlay.accept(caption("", 3, true)));
assert.equal(overlay.liveText(), "");
assert(!overlay.accept(caption("late", 4)));
console.log("Live captions remain independent of immediate final history and retain fixed lines.");
