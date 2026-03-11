import { type PlayerState } from "../utils/api.types";
import { type Card } from "../utils/api.types";
import { useState, useEffect, useRef } from "react";

// TODO - do we need to split the SCSS for this component?
import "./PlayerStateBox.scss";

const RESOURCES: { type: Card; label: string; className: string }[] = [
  { type: "WOOD", label: "Wood", className: "wood-cards" },
  { type: "BRICK", label: "Brick", className: "brick-cards" },
  { type: "SHEEP", label: "Sheep", className: "sheep-cards" },
  { type: "WHEAT", label: "Wheat", className: "wheat-cards" },
  { type: "ORE", label: "Ore", className: "ore-cards" },
];

const DEV_CARDS: {
  type: Card;
  label: string;
  shortLabel: string;
  className: string;
}[] = [
  {
    type: "VICTORY_POINT",
    label: "Victory Point",
    shortLabel: "VP",
    className: "dev-cards",
  },
  {
    type: "KNIGHT",
    label: "Knight",
    shortLabel: "KN",
    className: "dev-cards",
  },
  {
    type: "MONOPOLY",
    label: "Monopoly",
    shortLabel: "MO",
    className: "dev-cards",
  },
  {
    type: "YEAR_OF_PLENTY",
    label: "Year of Plenty",
    shortLabel: "YP",
    className: "dev-cards",
  },
  {
    type: "ROAD_BUILDING",
    label: "Road Building",
    shortLabel: "RB",
    className: "dev-cards",
  },
];

function ResourceCardItem({
  type,
  label,
  className,
  count,
  size,
  shortLabel,
}: {
  type: string;
  label: string;
  className: string;
  count: number;
  size?: "small" | "large";
  shortLabel?: string;
}) {
  const [animate, setAnimate] = useState(false);
  const prevCount = useRef(count);

  useEffect(() => {
    if (count !== prevCount.current) {
      setAnimate(true);
      const timer = setTimeout(() => setAnimate(false), 300); // 300ms matches animation duration
      return () => clearTimeout(timer);
    }
    prevCount.current = count;
  }, [count]);

  if (count === 0) return null;

  return (
    <div
      key={type}
      className={`${className} resource-card-container ${
        animate ? "animate-jump" : ""
      }`}
      title={shortLabel ? `${count} ${label} Card(s)` : undefined}
    >
      <div className="card-visual">
        <span className="count">{count}</span>
        {shortLabel && <span className="short-label">{shortLabel}</span>}
      </div>
      {size === "large" && <span className="label">{label}</span>}
    </div>
  );
}

export default function ResourceCards({
  playerState,
  playerKey,
  size = "small",
}: {
  playerState: PlayerState;
  playerKey: string;
  size?: "small" | "large";
}) {
  const amount = (card: Card) => playerState[`${playerKey}_${card}_IN_HAND`];
  return (
    <div className={`resource-cards ${size}`} title="Resource Cards">
      {RESOURCES.map(({ type, label, className }) => (
        <ResourceCardItem
          key={type}
          type={type}
          label={label}
          className={className}
          count={amount(type)}
          size={size}
        />
      ))}
      {DEV_CARDS.map(({ type, label, shortLabel, className }) => (
        <ResourceCardItem
          key={type}
          type={type}
          label={label}
          className={className}
          count={amount(type)}
          size={size}
          shortLabel={shortLabel}
        />
      ))}
    </div>
  );
}
