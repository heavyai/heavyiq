import React from "react";
import Header from "../components/Header";
import SendOutlinedIcon from "@mui/icons-material/SendOutlined";
import { useEffect, useRef, useState } from "react";
import { socket } from "../socket";
import ColoredCircleIcon from "../components/CircleIcon";
import {FinalThoughtStep, IntermediateStep, EventStep} from "../components/Step";

import {
  Stack,
  Box,
  Typography,
  FormControl,
  OutlinedInput,
  CircularProgress,
  List
} from "@mui/material";
import Answer from "../components/Answer";
import Steps from "../components/Steps";

const messageType = {
  answer: "answer",
  question: "question",
  steps: "steps",
};

const socketStatusType = {
  connected: "Connected",
  disconnected: "Disconnected",
  connecting: "Connecting",
};

const HomePage = () => {
  const inputRef = useRef();
  const chatWrapperRef = useRef();

  // enabled on question request and disabled once the answer arrives
  const [onRequest, setOnRequest] = useState(false);
  // storing question data
  const [question, setQuestion] = useState("");
  // storing final answer
  const [answer, setAnswer] = useState("");
  // list of messages to show
  const [messages, setMessages] = useState([]);
  // enabled and disabled upon llm start and ll end
  const [onStreaming, setOnStreaming] = useState(false);
  // enabled once the typewriter module finishes writing the content
  const [typingComplete, setTypingComplete] = useState(false);
  // whether to show or not the streaming content
  const [showStreamingAnswer, setShowStreamingAnswer] = useState(false);
  // used to store intermediate tokens
  const [content, setContent] = useState(""); // tokens
  // intermediate step items
  const [stepItems, setStepItems] = useState([
    // {
    //   thought:
    //     'Running SQL query... "SELECT STATE_NAME, (HISPANIC * 100.0 / POPULATION) AS percentage_hispanic FROM usa_states ORDER BY percentage_hispanic DESC LIMIT "5"',
    //   observation:
    //     "[('New Mexico', 44.9), ('California', 35.3), ('Texas', 33.4), ('Arizona', 26.9), ('Nevada', 23.9)]",
    // },
    // {
    //   thought: "You?",
    //   observation: "Error! Dont know what you mean."
    // }
    // or
    // {
    // "text": "LLM Started!"
    // }
  ]);
  // final thought
  const [finalThought, setFinalThought] = useState("");
  // socket status
  const [socketStatus, setSocketStatus] = useState(
    socketStatusType.disconnected
  );

  const onEnterPress = (e) => {
    if (e.keyCode === 13) {
      sendMessage();
    }
  };

  useEffect(() => {
    setTimeout(() => {
      chatWrapperRef.current.addEventListener("DOMNodeInserted", (e) => {
        e.currentTarget.scroll({
          top: e.currentTarget.scrollHeight,
          behavior: "smooth",
        });
      });
    }, 200);
  }, []);

  const addItemToSteps = (newItem) => {
    setStepItems((prevItems) => [...prevItems, newItem]);
  };

  useEffect(() => {
    // Event handler for receiving messages and appending
    // it to the existing messages list
    const handleAnswer = (data) => {
      setAnswer(data.data);
    };

    const handleSetSession = (data) => {
      sessionStorage.setItem("sessionID", data.data);
    };

    const handleStartToken = (data) => {
      setOnStreaming(true);
    };

    const handleNewToken = (data) => {
      setContent(data.data); // Set the updated content to trigger the effect
    };

    const handleEndToken = (data) => {
      setOnStreaming(false);
      document.querySelector(".Typewriter__cursor").style.display = "none";
      setOnRequest(false);
    };

    const handleIntermediateStep = (data) => {
      // we don't need to handle intermediate step since
      // socket callback returns the necessary data
      // addItemToSteps({
      //   thought: data["thought"],
      //   observation: data["observation"],
      // });
    };

    const handleFinalThought = (data) => {
      setFinalThought(data);
    };

    const handleCallbackEvent = ({ text, status, data }) => {
      addItemToSteps({
        text: text,
        status: status
      });
    };

    socket.on("answer", handleAnswer);

    // Event for receiving session id
    socket.on("setSession", handleSetSession);

    socket.on("startToken", handleStartToken);

    socket.on("newToken", handleNewToken);

    socket.on("endToken", handleEndToken);

    socket.on("intermediateStep", handleIntermediateStep);

    // calback events
    socket.on("llm_start", handleCallbackEvent);
    socket.on("llm_error", handleCallbackEvent);
    socket.on("llm_end", handleCallbackEvent);
    socket.on("chain_start", handleCallbackEvent);
    socket.on("chain_error", handleCallbackEvent);
    socket.on("chain_end", handleCallbackEvent);
    socket.on("tool_start", handleCallbackEvent);
    socket.on("tool_error", handleCallbackEvent);
    socket.on("tool_end", handleCallbackEvent);
    socket.on("agent_action", handleCallbackEvent);
    socket.on("agent_finish", handleCallbackEvent);

    socket.on("intermediateStep", handleIntermediateStep);

    socket.on("finalThought", handleFinalThought);

    socket.on("connect", () => {
      setSocketStatus(socketStatusType.connected);
      console.log("Connected to socket server");
    });

    socket.on("disconnect", () => {
      setSocketStatus(socketStatusType.disconnected);
      console.log("Disconnected from socket server");
    });

    socket.io.on("reconnect_attempt", (attempt) => {
      setSocketStatus(socketStatusType.connecting);
      console.log("Reconnecting to socket server...");
    });

    socket.on("connect_error", (error) => {
      setSocketStatus(socketStatusType.disconnected);
      console.log("Connection error:", error);
    });

    socket.on("connect_timeout", (timeout) => {
      setSocketStatus(socketStatusType.disconnected);
      console.log("Connection timeout:", timeout);
    });

    // Cleanup function to disconnect when component unmounts
    return () => {
      socket.off("answer", handleAnswer);
      socket.off("setSession", handleSetSession);
      socket.off("startToken", handleStartToken);
      socket.off("newToken", handleNewToken);
      socket.off("endToken", handleEndToken);
      socket.off("intermediateStep", handleIntermediateStep);
      socket.off("finalThought", handleFinalThought);
      socket.off("llm_start", handleCallbackEvent);
      socket.off("llm_error", handleCallbackEvent);
      socket.off("llm_end", handleCallbackEvent);
      socket.off("chain_start", handleCallbackEvent);
      socket.off("chain_error", handleCallbackEvent);
      socket.off("chain_end", handleCallbackEvent);
      socket.off("tool_start", handleCallbackEvent);
      socket.off("tool_error", handleCallbackEvent);
      socket.off("tool_end", handleCallbackEvent);
      socket.off("agent_action", handleCallbackEvent);
      socket.off("agent_finish", handleCallbackEvent);
    };
  }, []);

  useEffect(() => {
    if (answer && typingComplete) {
      if (stepItems.length > 0) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            type: messageType.steps,
            content: stepItems,
            finalThought: finalThought,
          },
        ]);
        setStepItems([]);
        setFinalThought("");
      }
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          type: messageType.answer,
          content: answer,
        },
      ]);
      setAnswer("");
      setTypingComplete(false);
      setContent("");
    }
  }, [answer, typingComplete, stepItems, finalThought]);

  useEffect(() => {
    socket.connect();

    return () => {
      socket.disconnect();
    };
  }, []);

  const sendMessage = () => {
    if (onRequest | onStreaming) return;
    if (question.trim() !== "") {
      // Emit a 'message' event to the server
      const sessionID = sessionStorage.getItem("sessionID");
      socket.emit("question", { question: question, sessionID: sessionID });
      setOnRequest(true);
      setQuestion("");
      setShowStreamingAnswer(true);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          type: messageType.question,
          content: question,
        },
      ]);
    }
  };

  const handleTypingComplete = () => {
    // Update parent state or perform any action
    if (!onRequest) {
      setTypingComplete(true);
      setShowStreamingAnswer(false);
    }
  };

  return (
    <Stack
      alignItems="center"
      justifyContent="space-between"
      direction="column"
      sx={{ height: "100%", width: "100%", minHeight: "100vh" }}
    >
      <Header bg borderBottom>
        <Box
          sx={{
            width: "100%",
            height: "100%",
            position: "relative",
            paddingX: 2,
            maxWidth: "md",
          }}
        >
          <Typography
            variant="h6"
            fontWeight="700"
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
          >
            Heavy AI Assistant
          </Typography>
        </Box>
      </Header>

      <Box
        ref={chatWrapperRef}
        sx={{
          height: "100%",
          position: "fixed",
          zIndex: 1,
          maxWidth: "md",
          width: "100%",
          overflowY: "auto",
          paddingTop: "60px",
          paddingBottom: "90px",
          "&::-webkit-scrollbar": {
            width: "0px",
          },
        }}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
            maxWidth: "md",
            width: "100%",
          }}
        >
          {messages.map((item, index) =>
            item.type === messageType.steps ? (
              <Steps
                key={index}
                stepItems={item.content}
                finalThought={item.finalThought}
              ></Steps>
            ) : (
              <Box key={index} padding={1}>
                <Box
                  sx={{
                    padding: 2,
                    bgcolor:
                      item.type === messageType.answer
                        ? "#2f2f2f"
                        : item.type === messageType.question
                        ? "#c956d1"
                        : "",
                    borderRadius: 3,
                  }}
                >
                  {item.content}
                </Box>
              </Box>
            )
          )}
          {stepItems && (
            <div>
              <Box height="auto">
                <List
                  sx={{
                    listStyleType: "disc",
                    listStylePosition: "inside",
                    display: "flex",
                    flexDirection: "column",
                    gap: 0,
                    // pl: 2
                  }}
                >
                  {stepItems.map((item, index) =>
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
                </List>
              </Box>
            </div>
          )}
          {showStreamingAnswer && (
            <Answer
              streamingInput={content}
              onTypingComplete={handleTypingComplete}
            />
          )}
        </Box>
      </Box>

      <Stack
        width="100%"
        alignItems="center"
        justifyContent="center"
        borderTop="1px solid #2c2c2c"
        bgcolor="#000"
        zIndex={3}
        direction="row"
        sx={{
          position: "sticky",
          bottom: 0,
        }}
      >
        <Box display="flex" alignItems="center">
          <ColoredCircleIcon
            color={
              socketStatus === socketStatusType.connected
                ? "success"
                : socketStatus === socketStatusType.connecting
                ? "warning"
                : "error"
            }
            fontSize="small"
          />
          <Box marginLeft={1}>{socketStatus}</Box>
        </Box>
        <Box padding={2} width="100%" maxWidth="md">
          <FormControl fullWidth variant="outlined">
            <OutlinedInput
              inputRef={inputRef}
              sx={{
                "& .MuiOutlinedInput-notchedOutline": {
                  border: "none",
                },
              }}
              endAdornment={
                onRequest ? (
                  <CircularProgress size="1.5rem" />
                ) : (
                  <SendOutlinedIcon onClick={sendMessage} />
                )
              }
              autoFocus
              disabled={onRequest}
              onKeyUp={onEnterPress}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask something..."
            />
          </FormControl>
        </Box>
      </Stack>
    </Stack>
  );
};

export default HomePage;
