import Header from "../components/Header";
import SendOutlinedIcon from "@mui/icons-material/SendOutlined";
import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { socket } from "../socket";

import {
  Stack,
  Box,
  Typography,
  FormControl,
  OutlinedInput,
  CircularProgress,
} from "@mui/material";
import Answer from "../components/Answer";

const messageType = {
  answer: "answer",
  question: "question",
};

const HomePage = () => {
  const inputRef = useRef();
  const chatWrapperRef = useRef();
  const navigate = useNavigate();

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

    socket.on("answer", handleAnswer);

    // Event for receiving session id
    socket.on("setSession", handleSetSession);

    socket.on("startToken", handleStartToken);

    socket.on("newToken", handleNewToken);

    socket.on("endToken", handleEndToken);

    // Cleanup function to disconnect when component unmounts
    return () => {
      socket.off("answer", handleAnswer);
      socket.off("setSession", handleSetSession);
      socket.off("startToken", handleStartToken);
      socket.off("newToken", handleNewToken);
      socket.off("endToken", handleEndToken);
    };
  }, []);

  useEffect(() => {
    if (answer && typingComplete) {
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
  }, [answer, typingComplete]);

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
          {messages.map((item, index) => (
            <Box key={index} padding={1}>
              <Box
                sx={{
                  padding: 2,
                  bgcolor: item.type === messageType.answer && "#2f2f2f",
                  borderRadius: 3,
                }}
              >
                {item.content}
              </Box>
            </Box>
          ))}
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
        sx={{
          position: "sticky",
          bottom: 0,
        }}
      >
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
