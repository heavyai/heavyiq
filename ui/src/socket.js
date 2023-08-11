import { io } from 'socket.io-client';

// "undefined" means the URL will be computed from the `window.location` object
const URL = process.env.NODE_ENV === 'production' ? undefined : process.env.REACT_APP_API_BASE_URL;

export const socket = io(URL, {autoConnect: false,
    reconnection: true,
    reconnectionAttempts: 3, // Adjust as needed
    reconnectionDelay: 1000,
    transports: ["websocket"],
}
);