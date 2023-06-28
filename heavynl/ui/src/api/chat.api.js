import axiosClient from "./client";

export const chatCompletion = async ({ prompt }) => {
  try {
    const data = {
        "tables": [
            "usa_states"
          ],
        "question": prompt,
        "session_id": "khsad"
    }
    const response = await axiosClient.post("question", data);

    return { response };
  } catch (err) {
    return { err };
  }
};