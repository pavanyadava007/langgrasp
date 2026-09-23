import { useEffect, useRef, useState } from "react";
import { Button } from "../../components/ui";
import { ApiError, api } from "../../lib/api";
import { ms } from "../../lib/format";
import { useStore } from "../../store/store";

// Speech to text, with one rule from the hazard analysis (FMEA H6): the transcript lands in the command box
// and nothing else happens. A person reads it, edits it if it is wrong, and presses Run. Nothing is ever
// executed straight off the microphone.

type State = "idle" | "recording" | "transcribing" | "unsupported";

export function Microphone() {
  const setState = useStore((s) => s.setState);
  const notify = useStore((s) => s.notify);
  const [state, setLocal] = useState<State>("idle");
  const [level, setLevel] = useState(0);
  const [lastLatency, setLastLatency] = useState<number | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);
  const raf = useRef<number | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") setLocal("unsupported");
    return () => stop(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function cleanup() {
    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    void audioCtx.current?.close().catch(() => undefined);
    audioCtx.current = null;
    setLevel(0);
  }

  function stop(silent = false) {
    if (recorder.current && recorder.current.state === "recording") {
      if (silent) recorder.current.onstop = null;
      recorder.current.stop();
    }
    cleanup();
  }

  async function start() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = s;
      const ctx = new AudioContext();
      audioCtx.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(s).connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
        setLevel(peak);
        raf.current = requestAnimationFrame(tick);
      };
      tick();

      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      const rec = new MediaRecorder(s, { mimeType: mime });
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      rec.onstop = async () => {
        cleanup();
        const blob = new Blob(chunks.current, { type: mime });
        if (blob.size < 500) {
          setLocal("idle");
          notify("warn", "That recording was too short to transcribe. Hold the button while you speak.");
          return;
        }
        setLocal("transcribing");
        try {
          const r = await api.stt(blob);
          setLastLatency(r.latency_ms);
          if (!r.normalized) {
            notify("warn", "Nothing was recognised in that clip. Try again, or type the command.");
          } else {
            setState({ command: r.normalized });
            notify("info", `Transcribed with faster-whisper ${r.model_size} on ${r.device}. Check it, then press Run: speech is never executed on its own.`);
          }
        } catch (e) {
          notify("error", e instanceof ApiError ? e.message : String(e));
        } finally {
          setLocal("idle");
        }
      };
      rec.start();
      recorder.current = rec;
      setLocal("recording");
    } catch (e) {
      setLocal("idle");
      notify("error", `The microphone could not be opened: ${e instanceof Error ? e.message : String(e)}. Type the command instead.`);
    }
  }

  if (state === "unsupported") {
    return (
      <Button disabled title="This browser has no MediaRecorder, or the page is not in a secure context. Over an SSH tunnel to localhost it is; over a plain LAN address it is not.">
        Mic unavailable
      </Button>
    );
  }

  return (
    <span className="inline-flex items-center gap-2">
      <Button
        onClick={() => (state === "recording" ? stop() : void start())}
        variant={state === "recording" ? "danger" : "subtle"}
        disabled={state === "transcribing"}
        title="Record a spoken command. The transcript goes into the box; you still press Run."
      >
        {state === "recording" ? "Stop" : state === "transcribing" ? "Transcribing" : "Speak"}
      </Button>
      {state === "recording" && (
        <span className="inline-flex items-center gap-1" aria-hidden="true">
          {[0, 1, 2, 3].map((i) => (
            <span key={i} className="w-1 rounded-s bg-estop" style={{ height: `${6 + Math.min(1, level * 2.5) * 16 * (i % 2 ? 0.7 : 1)}px` }} />
          ))}
        </span>
      )}
      {state === "recording" && <span className="sr-only">Recording</span>}
      {lastLatency !== null && state === "idle" && <span className="num text-xs text-fg-muted">transcribed in {ms(lastLatency)}</span>}
    </span>
  );
}
