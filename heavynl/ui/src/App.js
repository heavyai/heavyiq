import "react-toastify/dist/ReactToastify.css";


import HomePage from "./pages/HomePage";

import { createTheme, ThemeProvider } from "@mui/material/styles";
import { CssBaseline } from "@mui/material";
import { ToastContainer } from "react-toastify";
import { BrowserRouter, Route, Routes } from "react-router-dom";

function App() {
  const theme = createTheme({
    palette: { mode: "dark" }
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ToastContainer
        position="bottom-left"
        autoClose={4000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnHover
      />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={
              <HomePage />
          } />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
