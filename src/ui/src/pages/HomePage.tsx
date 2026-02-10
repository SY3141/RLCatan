import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@mui/material";
import { GridLoader } from "react-spinners";

import "./HomePage.scss";

export default function HomePage() {
  const [loading] = useState(false); // keep if you later want loading again
  const [numPlayers, setNumPlayers] = useState(2);
  const navigate = useNavigate();

  const goPlayVsBot = () => {
    navigate(`/bots/play?players=${numPlayers}`);
  };

  const goBattleBots = () => {
    navigate(`/bots/battle?players=${numPlayers}`);
  };

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>

      <div className="switchable">
        {!loading ? (
          <>
            <ul>
              <li>OPEN HAND</li>
              <li>NO CHOICE DURING DISCARD</li>
            </ul>

            <div className="player-count-selector">
              <div className="player-count-label">Number of Players</div>
              <div className="player-count-buttons">
                {[2, 3, 4].map((value) => (
                  <Button
                    key={value}
                    variant="contained"
                    onClick={() => setNumPlayers(value)}
                    className={`player-count-button ${
                      numPlayers === value ? "selected" : ""
                    }`}
                  >
                    {value} Players
                  </Button>
                ))}
              </div>
            </div>

            <Button variant="contained" color="primary" onClick={goPlayVsBot}>
              Play Against a Bot
            </Button>

            <Button variant="contained" color="secondary" onClick={goBattleBots}>
              Battle Two Bots
            </Button>
          </>
        ) : (
          <GridLoader className="loader" color="#ffffff" size={60} />
        )}
      </div>
    </div>
  );
}