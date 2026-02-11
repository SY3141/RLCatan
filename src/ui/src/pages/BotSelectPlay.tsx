import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@mui/material";
import { getBots, createGameConfigured, type BotMeta } from "../utils/apiClient";

export default function BotSelectPlay() {
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const numPlayers = useMemo(() => {
    const raw = params.get("players");
    const n = raw ? parseInt(raw, 10) : 2;
    return Number.isFinite(n) ? Math.min(4, Math.max(2, n)) : 2;
  }, [params]);

  const [bots, setBots] = useState<BotMeta[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const list = await getBots();
        list.sort((a, b) => b.elo - a.elo);
        setBots(list);
        setSelected(list[0]?.id ?? "");
      } catch (e: any) {
        setError(e?.message ?? "Failed to load bots");
      }
    })();
  }, []);

  const start = async () => {
  setError("");
    try {
      const opponentKey = bots.find((x) => x.id === selected)?.key;
      if (!opponentKey) throw new Error("Selected bot has no key");

      const gameId = await createGameConfigured({
        mode: "human_vs_bot",
        numPlayers,
        opponentKey,
      });

      navigate(`/games/${gameId}`);
    } catch (e: any) {
      setError(e?.response?.data?.error ?? e?.message ?? "Failed to start game");
    }
  };

  return (
    <div style={{ padding: 24, color: "#fff" }}>
      <h2>Play against a bot</h2>
      <div style={{ opacity: 0.8, marginBottom: 12 }}>Players: {numPlayers}</div>

      {error && <div style={{ color: "crimson", marginBottom: 12 }}>{error}</div>}

      <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 520 }}>
        {bots.map((b) => (
          <label key={b.id} style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              type="radio"
              name="bot"
              value={b.id}
              checked={selected === b.id}
              onChange={() => setSelected(b.id)}
            />
            <span style={{ flex: 1 }}>{b.name ?? b.id}</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>Elo {b.elo}</span>
          </label>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <Button variant="contained" disabled={!selected} onClick={start}>
          Start game
        </Button>
      </div>
    </div>
  );
}