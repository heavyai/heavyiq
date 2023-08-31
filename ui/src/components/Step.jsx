import React from "react";
import ArrowRightAltIcon from "@mui/icons-material/ArrowRightAlt";
import CheckIcon from "@mui/icons-material/Check";
import ClearIcon from "@mui/icons-material/Clear";
import InfoIcon from "@mui/icons-material/Info";
import {
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
} from "@mui/material";

const listItemStyle = {
  isplay: "flex",
  padding: 0,
  lineHeight: "1",
};

const Step = ({ text, textColor, iconColor, icon }) => {
  return (
    <ListItem sx={listItemStyle}>
      {icon && <ListItemIcon sx={{ color: iconColor }}>{icon}</ListItemIcon>}
      <ListItemText
        disableTypography
        primary={
          <Typography
            variant="body2"
            style={{ color: textColor, fontFamily: "Montserrat, sans-serif" }}
          >
            {text}
          </Typography>
        }
      />
    </ListItem>
  );
};

const EventStep = ({ text, status }) => {
  if (status === "info") {
    return (
      <Step
        text={text}
        textColor="yellow"
        iconColor={"yellow"}
        icon={<ArrowRightAltIcon />}
      ></Step>
    );
  }
  if (status === "success") {
    return (
      <Step
        text={text}
        textColor="green"
        iconColor={"green"}
        icon={<CheckIcon />}
      ></Step>
    );
  } else {
    // error
    return (
      <Step
        text={text}
        textColor="red"
        iconColor={"red"}
        icon={<ClearIcon />}
      ></Step>
    );
  }
};

const IntermediateStep = ({ text }) => {
  return (
    <Step
      text={text}
      textColor="blue"
      iconColor={"green"}
      icon={<CheckIcon />}
    ></Step>
  );
};

const FinalThoughtStep = ({ text }) => {
  return (
    <Step
      text={"Final Thought: " + text}
      textColor="pink"
      iconColor={"pink"}
      icon={<InfoIcon />}
    ></Step>
  );
};

export { EventStep, IntermediateStep, FinalThoughtStep };
