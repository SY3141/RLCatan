import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@mui/material";
import { getBots, createGameConfigured, type BotMeta } from "../utils/apiClient";

export default function BotSelectBattle() {
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const numPlayers = useMemo(() => {
    const raw = params.get("players");
    const n = raw ? parseInt(raw, 10) : 2;
    return Number.isFinite(n) ? Math.min(4, Math.max(2, n)) : 2;
  }, [params]);

  const [bots, setBots] = useState<BotMeta[]>([]);
  const [a, setA] = useState<string>("");
  const [b, setB] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const list = await getBots();
        list.sort((x, y) => y.elo - x.elo);
        setBots(list);
        setA(list[0]?.id ?? "");
        setB(list[1]?.id ?? list[0]?.id ?? "");
      } catch (e: any) {
        setError(e?.message ?? "Failed to load bots");
      }
    })();
  }, []);

  const start = async () => {
    setError("");
    try {
      const botAKey = bots.find((x) => x.id === a)?.key;
      const botBKey = bots.find((x) => x.id === b)?.key;
      if (!botAKey || !botBKey) throw new Error("Selected bots missing keys");

      const gameId = await createGameConfigured({
        mode: "bot_vs_bot",
        numPlayers,
        botAKey,
        botBKey,
      });

    navigate(`/games/${gameId}`);
    } catch (e: any) {
      setError(e?.response?.data?.error ?? e?.message ?? "Failed to start game");
    }
  };

  const valid = a && b && a !== b;

  return (
    <div style={{ padding: 24, color: "#fff" }}>
      <h2>Battle two bots</h2>
      <div style={{ opacity: 0.8, marginBottom: 12 }}>Players: {numPlayers}</div>

      {error && <div style={{ color: "crimson", marginBottom: 12 }}>{error}</div>}

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div>
          <div style={{ marginBottom: 6 }}>Bot A</div>
          <select value={a} onChange={(e) => setA(e.target.value)}>
            {bots.map((x) => (
              <option key={x.id} value={x.id}>
                {(x.name ?? x.id) + ` (Elo ${x.elo})`}
              </option>
            ))}
          </select>
        </div>

        <div>
          <div style={{ marginBottom: 6 }}>Bot B</div>
          <select value={b} onChange={(e) => setB(e.target.value)}>
            {bots.map((x) => (
              <option key={x.id} value={x.id}>
                {(x.name ?? x.id) + ` (Elo ${x.elo})`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {!valid && <div style={{ marginTop: 10, color: "crimson" }}>Pick two different bots.</div>}

      <div style={{ marginTop: 16 }}>
        <Button variant="contained" disabled={!valid} onClick={start}>
          Start battle
        </Button>
      </div>
    </div>
  );
}