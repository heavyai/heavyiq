import React from "react";
import { Box, List, ListItem, ListItemText, ListItemIcon } from "@mui/material";
import CheckIcon from "@mui/icons-material/Check";
import ClearIcon from "@mui/icons-material/Clear";
import InfoIcon from '@mui/icons-material/Info';

const Steps = ({ stepItems, finalThought }) => {
  return (
    <div>
      <Box height="auto">
        <List
          sx={{
            listStyleType: "disc",
            listStylePosition: "inside",
            display: "flex",
            flexDirection: "column",
            gap: 0,
          }}
        >
          {stepItems.map((item, index) => (
            <React.Fragment key={`step-item-${index}`}>
              <ListItem sx={{ color: "yellow", display: "flex" }}>
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
                  display: "flex",
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
