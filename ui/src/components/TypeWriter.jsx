import React, { useEffect, useRef } from "react";
import Typewriter from "typewriter-effect";

const TypeWriter = ({ streamingInput, options, onTypingComplete }) => {
  const typewriterRef = useRef(null);

  useEffect(() => {
    if (typewriterRef.current) {
      typewriterRef.current
        .typeString(streamingInput)
        .callFunction(() => {
          if (onTypingComplete) {
            onTypingComplete();
          }
        })
        .start();
    }
  }, [streamingInput, onTypingComplete]);

  return (
    <Typewriter
      onInit={(typewriter) => (typewriterRef.current = typewriter)}
      options={options}
    />
  );
};

export default TypeWriter;
