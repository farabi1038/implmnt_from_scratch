package main

// TCP Echo Server - Version 1
// Goal: Accept multiple clients, echo back what they send

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"time"
)
var (
    totalConnections  int64
    activeConnections int64
)


func main() {
	// Step 1 - Create a TCP listener on port 8080
	// In Java this would be: new ServerSocket(8080)
	listener, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Server listening on :8080")

	// Step 2 - Accept loop (infinite loop waiting for clients)
	// In Java: while(true) { socket = serverSocket.accept(); }
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Println(err) // log.Println, not Fatal - we want to keep running
			continue
		}
		// Handle each client in a goroutine
		go handleConnection(conn)
	}
}

// handleConnection processes a single client connection
// TODO: Step 4 - Read from client, echo back, handle errors
func handleConnection(conn net.Conn) {
	clientAddr := conn.RemoteAddr().String()
	log.Printf("Client Connected : %s",clientAddr)
	// In Java this would be:
	// BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
	// PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
	// String line; while((line = in.readLine()) != null) { out.println(line); }
	reader := bufio.NewReader(conn)
	var disconnectReason string
	for {
		// Reset deadline before each read - client has 30s to send next message
		conn.SetDeadline(time.Now().Add(30 * time.Second))
		line, err := reader.ReadString('\n')

		if err != nil {
			// Determine why we disconnected
			if os.IsTimeout(err) {
				disconnectReason = "timeout (30s idle)"
			} else {
				disconnectReason = err.Error() // e.g., "EOF" when client closes
			}
			break
		}
		conn.Write([]byte(line))
	}
	log.Printf("Client Disconnected : %s, Reason : %s", clientAddr, disconnectReason)
	conn.Close()
}
