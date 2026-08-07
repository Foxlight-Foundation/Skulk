/**
 * Code narration for the voice loop (#769): while a fenced code block is
 * streaming during live generation, the assistant voice acknowledges the
 * code instead of leaving dead air, with an opener when the fence starts,
 * occasional fillers while it streams, and a closer when it ends.
 *
 * The narrator is a pure state machine driven by the chat streaming loop;
 * it never touches audio itself. The essential design rule is that fillers
 * are silence fillers, not schedule items: one only fires when the sentence
 * queue is empty AND the playback buffer is nearly dry, so narration can
 * never stack behind pending prose or play back-to-back on a fast stream.
 * Replayed messages never construct a narrator, so playback stays silent
 * about code by construction.
 */

/** Phrase sets for one narrated response. Text is already localized. */
export interface CodeNarrationPhrases {
  openers: readonly string[];
  fillers: readonly string[];
  closers: readonly string[];
}

/** Signals sampled by the streaming loop on each delta. */
export interface CodeNarrationSignals {
  /** Whether the retained speech tail currently holds an unclosed fence. */
  fenceOpen: boolean;
  /** Whether the sentence queue has nothing pending or synthesizing. */
  queueStarved: boolean;
  /** Seconds of decoded audio still queued for playback. */
  bufferedSeconds: number;
}

export interface CodeNarratorOptions {
  /** Minimum spacing between narration utterances. */
  minimumGapMs?: number;
  /** A fence reopening within this window continues silently (no new
   *  opener, and the pending closer is cancelled). */
  reopenDebounceMs?: number;
  /** Fillers per fence beyond which the narrator stays quiet. */
  maximumFillersPerFence?: number;
  /** Playback buffer level below which the voice counts as running dry. */
  starvedBufferSeconds?: number;
  /** Injectable clock for tests. */
  now?: () => number;
  /** Injectable randomness for tests. */
  random?: () => number;
}

/** Emits at most one narration utterance per update; null means silence. */
export class CodeNarrator {
  private readonly minimumGapMs: number;
  private readonly reopenDebounceMs: number;
  private readonly maximumFillersPerFence: number;
  private readonly starvedBufferSeconds: number;
  private readonly now: () => number;
  private readonly random: () => number;

  private fenceOpen = false;
  private openerPlayed = false;
  private fillersThisFence = 0;
  private lastUtteranceAt = Number.NEGATIVE_INFINITY;
  private pendingCloseSince: number | null = null;
  private lastPhrase = '';

  constructor(
    private readonly phrases: CodeNarrationPhrases,
    options: CodeNarratorOptions = {},
  ) {
    this.minimumGapMs = options.minimumGapMs ?? 7000;
    this.reopenDebounceMs = options.reopenDebounceMs ?? 2500;
    this.maximumFillersPerFence = options.maximumFillersPerFence ?? 3;
    this.starvedBufferSeconds = options.starvedBufferSeconds ?? 0.75;
    this.now = options.now ?? (() => Date.now());
    this.random = options.random ?? Math.random;
  }

  /** Advance the state machine with fresh stream signals. */
  update(signals: CodeNarrationSignals): string | null {
    const timestamp = this.now();

    if (signals.fenceOpen && !this.fenceOpen) {
      this.fenceOpen = true;
      if (this.pendingCloseSince !== null) {
        // Adjacent fence: the block continued, so swallow both the pending
        // closer and a fresh opener rather than chattering between them.
        this.pendingCloseSince = null;
        return null;
      }
      this.fillersThisFence = 0;
      this.openerPlayed = true;
      this.lastUtteranceAt = timestamp;
      return this.pick(this.phrases.openers);
    }

    if (!signals.fenceOpen && this.fenceOpen) {
      this.fenceOpen = false;
      if (this.openerPlayed) this.pendingCloseSince = timestamp;
      return null;
    }

    if (
      this.pendingCloseSince !== null
      && timestamp - this.pendingCloseSince >= this.reopenDebounceMs
    ) {
      this.pendingCloseSince = null;
      this.openerPlayed = false;
      this.lastUtteranceAt = timestamp;
      return this.pick(this.phrases.closers);
    }

    if (
      this.fenceOpen
      && this.openerPlayed
      && signals.queueStarved
      && signals.bufferedSeconds < this.starvedBufferSeconds
      && timestamp - this.lastUtteranceAt >= this.minimumGapMs
      && this.fillersThisFence < this.maximumFillersPerFence
    ) {
      this.fillersThisFence += 1;
      this.lastUtteranceAt = timestamp;
      return this.pick(this.phrases.fillers);
    }

    return null;
  }

  /** Flush the closer at end of stream, including an unterminated fence. */
  finish(): string | null {
    const shouldClose = this.openerPlayed && (this.fenceOpen || this.pendingCloseSince !== null);
    this.fenceOpen = false;
    this.pendingCloseSince = null;
    this.openerPlayed = false;
    return shouldClose ? this.pick(this.phrases.closers) : null;
  }

  private pick(list: readonly string[]): string | null {
    if (list.length === 0) return null;
    const candidates = list.length > 1 ? list.filter((p) => p !== this.lastPhrase) : list;
    const choice = candidates[Math.floor(this.random() * candidates.length)] ?? candidates[0];
    this.lastPhrase = choice;
    return choice;
  }
}
