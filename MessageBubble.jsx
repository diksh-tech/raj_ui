import React from "react";

export default function MessageBubble({ role, content, time }) {
  return (
    <div className={`msg-row ${role === "user" ? "is-user" : "is-assistant"}`}>
      {role === "assistant" && <div className="msg-avatar assistant">🤖</div>}

      <div className={`msg-bubble ${role}`}>
        <p>{content}</p>
        <span className="msg-time">{time}</span>
      </div>

      {role === "user" && <div className="msg-avatar user">🧑</div>}
    </div>
  );
}
