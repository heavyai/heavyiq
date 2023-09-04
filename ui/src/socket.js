import { io } from 'socket.io-client';

// "undefined" means the URL will be computed from the `window.location` object
const URL = process.env.NODE_ENV === 'production' ? undefined : process.env.REACT_APP_API_BASE_URL;

export const socket = io(URL, {autoConnect: false,
    reconnection: true,
    reconnectionAttempts: 10, // Adjust as needed
    reconnectionDelay: 2000,
    transports: ["websocket"],
}
);