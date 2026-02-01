# TCP + HTTP Server From Scratch - Learning Journal

> A hands-on journey building network servers in Go, documented step-by-step.
> Built by: Ibne Farabi Shihab
> Started: January 31, 2026

## 🎯 Project Goal
Build a production-quality TCP and HTTP server from scratch to deeply understand:
- Network programming fundamentals
- Concurrent connection handling
- Failure modes and resilience
- Observability (metrics, logging)

## 📚 My Background
- Strong Java experience
- New to Go (learning syntax as I build)

---

## Week 1: TCP Server Foundation

### Day 1-2: Multi-Client TCP Server

#### Concepts Learned

**1. TCP Basics (vs Java)**

In Java, you'd write:
```java
ServerSocket serverSocket = new ServerSocket(8080);
while (true) {
    Socket client = serverSocket.accept();  // blocks here
    new Thread(() -> handleClient(client)).start();
}
```

In Go, it's similar but with goroutines:
```go
listener, _ := net.Listen("tcp", ":8080")
for {
    conn, _ := listener.Accept()  // blocks here
    go handleClient(conn)         // goroutine, not thread!
}
```

**Key difference:** Goroutines are ~2KB of stack (vs ~1MB for Java threads). You can spawn thousands easily.

**2. Goroutines vs Java Threads**

| Java Thread | Go Goroutine |
|-------------|--------------|
| OS-level thread | Green thread (Go runtime manages) |
| ~1MB stack | ~2KB stack (grows as needed) |
| `new Thread().start()` | `go functionName()` |
| Expensive to create | Cheap, create thousands |

**3. Error Handling in Go**

Go has no exceptions. Functions return errors explicitly:
```go
conn, err := listener.Accept()
if err != nil {
    // handle error
}
```

This feels verbose at first, but it forces you to think about every failure point.

---

#### What I Built

**Version 1: Basic Echo Server**
- File: `cmd/server/main.go`
- Status: ✅ Complete!

**Features working:**
- Listens on port 8080
- Accepts multiple clients simultaneously
- Each client handled in its own goroutine
- Echoes back whatever client sends
- 30-second idle timeout (auto-disconnects inactive clients)
- Detailed logging (connect/disconnect with reasons)
- Connection metrics (active count, total count)

#### Challenges & Solutions

**Challenge 1: Port already in use**
When server crashes or doesn't shut down cleanly, the port stays reserved.
```
listen tcp :8080: bind: address already in use
```
**Solution:** Kill the old process with `lsof -ti :8080 | xargs kill -9`

**Challenge 2: Understanding `:=` vs `=`**
Go has two assignment operators which confused me initially.
- `:=` declares AND assigns (creates new variable)
- `=` assigns only (variable must already exist)

#### Questions I Had

**Q: Why does Go put types after variable names?**
A: It reads more naturally for complex types. `var data map[string][]int` reads as "data is a map of string to slice of int". Also, type inference with `:=` means you rarely write types explicitly anyway.

**Q: Why no exceptions in Go?**
A: Go forces explicit error handling. Every function that can fail returns an error. This makes you think about failure at every step - annoying at first, but leads to more robust code.

---

### Day 3-4: Stress Testing & Edge Cases
*[Coming soon]*

### Day 5: Failure Analysis
*[Coming soon]*

### Weekend: Metrics & Chaos Testing
*[Coming soon]*

---

## 💡 Key Insights

1. **Goroutines are incredibly cheap** - In Java I'd worry about thread pool sizing. In Go, just spawn a goroutine per connection. The runtime handles it.

2. **Explicit error handling changes how you think** - No try/catch means you handle errors immediately where they occur. Verbose but clear.

3. **Case sensitivity matters everywhere** - `log.Fatal` vs `log.fatal`, `String` vs `string`. Coming from Java, this tripped me up multiple times.

4. **Timeouts should reset on activity** - Setting a deadline once at connection start is wrong. Reset it after each successful read for "idle timeout" behavior.

## 🐛 Bugs I Created & Fixed

### Bug #1: Case Sensitivity in Go
**What I wrote:**
```go
if err != nill { log.fatal(err) }
fmt.println("Server listening")
```

**The fix:**
```go
if err != nil { log.Fatal(err) }
fmt.Println("Server listening")
```

**Lesson learned:** Go is case-sensitive. Exported (public) functions start with uppercase. `nil` not `nill`!

### Bug #2: String type is lowercase
**What I wrote:**
```go
var disconnectReason String
```

**The fix:**
```go
var disconnectReason string
```

**Lesson learned:** Go primitives are lowercase (`string`, `int`, `bool`), unlike Java's wrapper classes (`String`, `Integer`, `Boolean`).

### Bug #3: Wrong assignment operator
**What I wrote:**
```go
conn, err = listener.Accept()  // inside loop, first use
```

**The fix:**
```go
conn, err := listener.Accept()  // := for new variables
```

**Lesson learned:** Use `:=` when declaring new variables, `=` only for reassignment.

### Bug #4: Atomic function case sensitivity
**What I wrote:**
```go
atomic.Addint64(&activeConnections, 1)
```

**The fix:**
```go
atomic.AddInt64(&activeConnections, 1)
```

**Lesson learned:** Even within function names, case matters. `Int64` not `int64`.

## 📖 Resources Used

- [Go Documentation](https://golang.org/doc/)
- [Network Programming with Go](https://tumregels.github.io/Network-Programming-with-Go/)
