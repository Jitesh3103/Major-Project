// ==============================
// Toggle Between Login & Register
// ==============================
const container = document.querySelector('.container');
const registerBtn = document.querySelector('.register-btn');
const loginBtn = document.querySelector('.login-btn');

registerBtn.addEventListener('click', () => {
    container.classList.add('active');
});

loginBtn.addEventListener('click', () => {
    container.classList.remove('active');
});

// ==============================
// Auto-open login/register based on hash in URL
// ==============================
window.addEventListener("load", () => {
  if (window.location.hash === "#register") {
    document.querySelector(".register-btn").click();
  } else if (window.location.hash === "#login") {
    document.querySelector(".login-btn").click();
  }
});

// ==============================
// Simple "already registered" demo storage
// ==============================
let registeredUsers = [];

// ==============================
// Handle Registration
// ==============================
document.getElementById("registerForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const username = document.getElementById("regUsername").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;
  const confirmPassword = document.getElementById("regConfirmPassword").value;
  const msg = document.getElementById("registerMessage");

  if (password !== confirmPassword) {
    msg.textContent = "❌ Passwords do not match!";
    msg.className = "msg error";
    return;
  }

  if (registeredUsers.includes(email)) {
    msg.textContent = "⚠️ You are already registered!";
    msg.className = "msg warning";
  } else {
    registeredUsers.push(email);
    msg.textContent = "✅ Successfully registered!";
    msg.className = "msg success";
    document.getElementById("registerForm").reset();
  }
});

// ==============================
// Handle Login
// ==============================
document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;
  const msg = document.getElementById("loginMessage");

  // Simple demo validation (just checks if registered email exists)
  if (registeredUsers.length === 0) {
    msg.textContent = "⚠️ No registered users yet!";
    msg.className = "msg warning";
    return;
  }

  // For demo: accept any registered email as username
  if (registeredUsers.includes(username)) {
    msg.textContent = "✅ Login successful!";
    msg.className = "msg success";
    document.getElementById("loginForm").reset();
  } else {
    msg.textContent = "❌ Invalid username or password!";
    msg.className = "msg error";
  }
});

// ==============================
// Handle Registration with MongoDB
// ==============================
document.getElementById("registerForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const username = document.getElementById("regUsername").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;
  const confirmPassword = document.getElementById("regConfirmPassword").value;
  const msg = document.getElementById("registerMessage");

  if (password !== confirmPassword) {
    msg.textContent = "❌ Passwords do not match!";
    msg.className = "msg error";
    return;
  }

  try {
    const res = await fetch("http://localhost:5000/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    const data = await res.json();
    msg.textContent = data.message;
    msg.className = res.ok ? "msg success" : "msg error";

    if (res.ok) document.getElementById("registerForm").reset();
  } catch (err) {
    msg.textContent = "❌ Network error!";
    msg.className = "msg error";
  }
});

// ==============================
// Handle Login with MongoDB
// ==============================
document.getElementById("loginForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;
  const msg = document.getElementById("loginMessage");

  try {
    const res = await fetch("http://localhost:5000/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();
    msg.textContent = data.message;
    msg.className = res.ok ? "msg success" : "msg error";

    if (res.ok) {
      // ✅ Redirect to main page
      window.location.href = "main.html";
    }
  } catch (err) {
    msg.textContent = "❌ Network error!";
    msg.className = "msg error";
  }
});
