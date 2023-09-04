import React from "react";
import { Box } from "@mui/material";
import TypeWriter from "./TypeWriter";

const Answer = ({ streamingInput, onTypingComplete }) => {
  const typewriterOptions = {
    delay: 50,
  };

  return (
    <Box padding={1}>
      <Box
        sx={{
          padding: 2,
          bgcolor: "#2f2f2f",
          borderRadius: 3,
        }}
      >
        <TypeWriter
          streamingInput={streamingInput}
          options={typewriterOptions}
          onTypingComplete={onTypingComplete}
        />
      </Box>
    </Box>
  );
};

export default Answer;
