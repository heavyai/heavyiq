import React, { useState } from "react";
import {
  Box,
  List,
  IconButton
} from "@mui/material";
import { ExpandMore, ExpandLess } from "@mui/icons-material";
import {FinalThoughtStep, IntermediateStep, EventStep} from "./Step";

const Steps = ({ stepItems, finalThought }) => {
  const [expanded, setExpanded] = useState(false);

  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  return (
    <div style={{ display: "flex", flexDirection: "row" }}>
      <div style={{ paddingTop: 15 }}>
        <IconButton onClick={toggleExpand}>
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </div>
      <Box>
        <List
          sx={{
            listStyleType: "disc",
            listStylePosition: "inside",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            transition: "display 0.3s ease",
          }}
        >
          {expanded &&
            stepItems.map((item, index) =>
              "thought" in item ? (
                <React.Fragment key={`step-item-${index}`}>
                  <IntermediateStep text={"Thought: " + item.thought}></IntermediateStep>
                  <IntermediateStep text={"Observation: " + item.observation}></IntermediateStep>
                </React.Fragment>
              ) : (
                <EventStep key={`step-item-${index}`} text={item.text} status={item.status}></EventStep>
              )
            )}
          {finalThought && (
            <FinalThoughtStep text={finalThought}></FinalThoughtStep>
          )}
          {!expanded && !finalThought && stepItems.length > 0 && (
            <EventStep text={stepItems[stepItems.length-1].text} status={stepItems[stepItems.length-1].status}></EventStep>
          )}
        </List>
      </Box>
    </div>
  );
};

export default Steps;
