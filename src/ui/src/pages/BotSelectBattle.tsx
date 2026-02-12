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

  const valid = a && b;

  const dropdownStyle = {
    padding: 12,
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#1c1c1c",
    color: "#fff",
    width: 250,
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
        Battle Two Bots
      </h1>
      <div style={{ opacity: 0.8, marginBottom: 12, textAlign: "center" }}>
        Players: {numPlayers}
      </div>

      {error && (
        <div style={{ color: "crimson", marginBottom: 12, textAlign: "center" }}>
          {error}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", gap: 32, marginBottom: 24 }}>
        {/* Bot A */}
        <div>
          <div style={{ marginBottom: 6, textAlign: "center", color: "#ffccbc" }}>Bot A</div>
          <select value={a} onChange={(e) => setA(e.target.value)} style={dropdownStyle}>
            {bots.map((x) => (
              <option key={x.id} value={x.id}>
                {(x.name ?? x.id) + ` (Elo ${x.elo})`}
              </option>
            ))}
          </select>
        </div>

        {/* Bot B */}
        <div>
          <div style={{ marginBottom: 6, textAlign: "center", color: "#ffccbc" }}>Bot B</div>
          <select value={b} onChange={(e) => setB(e.target.value)} style={dropdownStyle}>
            {bots.map((x) => (
              <option key={x.id} value={x.id}>
                {(x.name ?? x.id) + ` (Elo ${x.elo})`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {!valid && (
        <div style={{ marginTop: 10, color: "crimson", textAlign: "center" }}>
          Pick two bots.
        </div>
      )}

      <div style={{ textAlign: "center", marginTop: 16 }}>
        <Button
          variant="contained"
          color="secondary"
          disabled={!valid}
          onClick={start}
          style={{
            padding: "12px 24px",
            fontSize: "1rem",
            borderRadius: 8,
          }}
        >
          Start Battle
        </Button>
      </div>
    </div>
  );
}
