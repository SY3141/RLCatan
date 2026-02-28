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

  const dropdownStyle = {
    padding: 12,
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#1c1c1c",
    color: "#fff",
    width: 300,
    fontSize: 18,
    maxHeight: 220,
    overflowY: "auto" as const,
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        color: "#fff",
        backgroundColor: "#121212",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ fontSize: "3rem", color: "#ffffff", marginBottom: 24 }}>
        Play Against a Bot
      </h1>
      <div style={{ opacity: 0.8, marginBottom: 12, textAlign: "center" }}>
        Players: {numPlayers}
      </div>

      {error && (
        <div style={{ color: "crimson", marginBottom: 12, textAlign: "center" }}>
          {error}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 24 }}>
        <select value={selected} onChange={(e) => setSelected(e.target.value)} style={dropdownStyle}>
          {bots.map((b) => (
            <option key={b.id} value={b.id}>
              {(b.name ?? b.id) + ` (Elo ${b.elo})`}
            </option>
          ))}
        </select>
      </div>

      <div style={{ textAlign: "center" }}>
        <Button
          variant="contained"
          color="secondary"
          disabled={!selected}
          onClick={start}
          style={{
            padding: "12px 24px",
            fontSize: "1rem",
            borderRadius: 8,
          }}
        >
          Start Game
        </Button>
      </div>
    </div>
  );
}
