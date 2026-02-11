import axios from "axios";

import { API_URL } from "../configuration";
import type { Color, GameAction, GameState } from "./api.types";

type Player = "HUMAN" | "RANDOM" | "CATANATRON";
export type StateIndex = number | `${number}` | "latest";

export async function createGame(players: Player[]) {
  const response = await axios.post(API_URL + "/api/games", { players });
  return response.data.game_id;
}

export async function getState(
  gameId: string,
  stateIndex: StateIndex = "latest"
): Promise<GameState> {
  const response = await axios.get(
    `${API_URL}/api/games/${gameId}/states/${stateIndex}`
  );
  return response.data;
}

/** action=undefined means bot action */
export async function postAction(gameId: string, action?: GameAction) {
  const response = await axios.post<GameState>(
    `${API_URL}/api/games/${gameId}/actions`,
    action
  );
  return response.data;
}

export type MCTSProbabilities = {
  [K in Color]: number;
};

type MCTSSuccessBody = {
  success: true;
  probabilities: MCTSProbabilities;
  state_index: number;
};
type MCTSErrorBody = {
  success: false;
  error: string;
  trace: string;
};

export async function getMctsAnalysis(
  gameId: string,
  stateIndex: StateIndex = "latest"
) {
  try {
    console.log("Getting MCTS analysis for:", {
      gameId,
      stateIndex,
      url: `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`,
    });

    if (!gameId) {
      throw new Error("No gameId provided to getMctsAnalysis");
    }

    const response = await axios.get<MCTSSuccessBody | MCTSErrorBody>(
      `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`
    );

    console.log("MCTS analysis response:", response.data);
    return response.data;
  } catch (error: any) {
    // AxiosResponse<MCTSErrorBody>
    console.error("MCTS analysis error:", {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      stack: error.stack,
    });
    throw error;
  }
}

export type PlayerKey = string;

export type BotMeta = {
  id: string;
  name?: string;
  elo: number;
  key: PlayerKey;
};

export async function getBots(): Promise<BotMeta[]> {
  const response = await axios.get(API_URL + "/api/bots");
  return response.data;
}

export type CreateGameConfig =
  | { mode: "human_vs_bot"; numPlayers: number; opponentKey: PlayerKey }
  | { mode: "bot_vs_bot"; numPlayers: number; botAKey: PlayerKey; botBKey: PlayerKey };

export async function createGameConfigured(cfg: CreateGameConfig) {
  let players: PlayerKey[] = [];

  if (cfg.mode === "human_vs_bot") {
    players = ["HUMAN", ...Array(Math.max(0, cfg.numPlayers - 1)).fill(cfg.opponentKey)];
  } else {
    const pair: PlayerKey[] = [cfg.botAKey, cfg.botBKey];
    players = Array.from({ length: cfg.numPlayers }, (_, i) => pair[i % 2]); // A,B,A,B...
  }

  const response = await axios.post(API_URL + "/api/games", { players });
  return response.data.game_id;
}
