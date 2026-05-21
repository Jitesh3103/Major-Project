const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bcrypt = require("bcrypt");
const bodyParser = require("body-parser");

const app = express();
app.use(cors());
app.use(bodyParser.json());

// ✅ Connect MongoDB
mongoose.connect("mongodb://127.0.0.1:27017/loginApp", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => console.log("MongoDB Connected"))
  .catch(err => console.log(err));

// ✅ User Schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

const User = mongoose.model("User", userSchema);

// ==============================
// Register Route
// ==============================
app.post("/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    const existing = await User.findOne({ email });
    if (existing) {
      return res.status(400).json({ message: "⚠️ Email already registered!" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = new User({ username, email, password: hashedPassword });
    await newUser.save();

    res.json({ message: "✅ Registration successful!" });
  } catch (err) {
    res.status(500).json({ message: "❌ Server Error" });
  }
});

// ==============================
// Login Route
// ==============================
app.post("/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    const user = await User.findOne({ username });
    if (!user) return res.status(400).json({ message: "❌ User not found!" });

    const validPass = await bcrypt.compare(password, user.password);
    if (!validPass) return res.status(400).json({ message: "❌ Invalid password!" });

    res.json({ message: "✅ Login successful!", redirect: "/main.html" });
  } catch (err) {
    res.status(500).json({ message: "❌ Server Error" });
  }
});

// Start Server
app.listen(5000, () => console.log("Server running on http://localhost:5000"));
