import React, { useState } from "react";
import {
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
} from "@mui/material";
import CheckIcon from "@mui/icons-material/Check";
import ClearIcon from "@mui/icons-material/Clear";
import InfoIcon from "@mui/icons-material/Info";
import { ExpandMore, ExpandLess } from "@mui/icons-material";

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
          {stepItems.map((item, index) => (
            <React.Fragment key={`step-item-${index}`}>
              <ListItem
                sx={{ color: "yellow", display: expanded ? "flex" : "none" }}
              >
                <ListItemIcon sx={{ color: "green" }}>
                  <CheckIcon />
                </ListItemIcon>
                <ListItemText
                  primary={"Thought: " + item.thought}
                  primaryTypographyProps={{
                    sx: { fontSize: 13, fontWeight: "bold" },
                  }}
                />
              </ListItem>
              <ListItem
                sx={{
                  color: item.observation.startsWith("Error") ? "red" : "green",
                  display: expanded ? "flex" : "none",
                }}
              >
                {item.observation.startsWith("Error") ? (
                  <ListItemIcon sx={{ color: "red" }}>
                    <ClearIcon />
                  </ListItemIcon>
                ) : (
                  <ListItemIcon sx={{ color: "green" }}>
                    <CheckIcon />
                  </ListItemIcon>
                )}

                <ListItemText
                  primary={"Observation: " + item.observation}
                  primaryTypographyProps={{
                    sx: { fontSize: 13, fontWeight: "bold" },
                  }}
                />
              </ListItem>
            </React.Fragment>
          ))}
          {finalThought && (
            <ListItem sx={{ color: "pink", display: "flex" }}>
              <ListItemIcon sx={{ color: "pink" }}>
                <InfoIcon />
              </ListItemIcon>
              <ListItemText
                primary={"Final Thought: " + finalThought}
                primaryTypographyProps={{
                  sx: { fontSize: 14, fontWeight: "bold" },
                }}
              />
            </ListItem>
          )}
        </List>
      </Box>
    </div>
  );
};

export default Steps;
