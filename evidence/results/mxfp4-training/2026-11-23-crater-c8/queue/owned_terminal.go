// Narrow adapter for the documented Crater Web Terminal protocol on one owned job.
// Authentication remains inside the official CLI session library; never log tokens.
package main

import (
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "net/http"
    "net/url"
    "os"
    "regexp"
    "strconv"
    "strings"
    "time"

    "github.com/gorilla/websocket"
    "github.com/raids-lab/crater/cli/internal/api"
    "github.com/raids-lab/crater/cli/internal/session"
)

const platform = "https://gpu.act.buaa.edu.cn"
const job = "sg-jincy26-261124-4abd5"
const pod = job + "-default0-0"

func fail(message string) { fmt.Fprintln(os.Stderr, message); os.Exit(1) }

func main() {
    if len(os.Args) != 3 { fail("usage: owned-terminal SCRIPT EXPECTED_SHA256") }
    source, err := os.ReadFile(os.Args[1])
    if err != nil || len(source) > 65536 { fail("script unavailable or too large") }
    digest := sha256.Sum256(source)
    if hex.EncodeToString(digest[:]) != os.Args[2] { fail("script SHA mismatch") }
    if time.Now().After(time.Date(2026,11,24,4,51,24,0,time.UTC)) { fail("original deadline reached") }
    state, err := session.LoadState()
    if err != nil { fail("native session unavailable") }
    if state.ActiveContext.Username != "jincy26" || state.ActiveContext.PlatformURL != platform {
        fail("unexpected active identity or platform")
    }
    token, err := session.LoadToken(state, state.ActiveContext)
    if err != nil { fail("native session token unavailable") }
    client := api.NewClient(platform).SetToken(token)
    detail, err := client.GetJob(job)
    if err != nil { fail("authenticated owned-job read failed") }
    encoded, err := json.Marshal(detail)
    if err != nil { fail("job metadata unavailable") }
    var identity map[string]any
    if json.Unmarshal(encoded, &identity) != nil { fail("job metadata invalid") }
    for key, expected := range map[string]string{"jobName":job,"username":"jincy26","namespace":"crater-workspace","createdAt":"2026-11-23T20:25:26Z","status":"Running"} {
        if identity[key] != expected { fail("owned-job identity or state changed: "+key) }
    }
    endpoint := "wss://gpu.act.buaa.edu.cn/api/v1/websocket/namespaces/crater-workspace/pods/"+pod+"/containers/custom/terminal"
    dialer := *websocket.DefaultDialer
    dialer.HandshakeTimeout = 10*time.Second
    // Match the official frontend WebSocket authentication format; never log this URL.
    endpoint += "?"+url.Values{"token":[]string{token}}.Encode()
    headers := http.Header{"Origin":[]string{platform}}
    socket, response, err := dialer.Dial(endpoint, headers)
    token = ""
    if err != nil {
        status := 0
        if response != nil { status=response.StatusCode; response.Body.Close() }
        fail(fmt.Sprintf("documented terminal connection failed; HTTP status %d",status))
    }
    defer socket.Close()
    socket.SetReadLimit(1<<20)
    socket.SetReadDeadline(time.Now().Add(45*time.Second))
    socket.SetWriteDeadline(time.Now().Add(10*time.Second))
    if socket.WriteJSON(map[string]any{"op":"resize","cols":120,"rows":40}) != nil { fail("terminal initial resize failed") }
    send := func(data string) {
        socket.SetWriteDeadline(time.Now().Add(10*time.Second))
        if socket.WriteJSON(map[string]string{"op":"stdin","data":data}) != nil { fail("terminal input failed; inspect before retry") }
    }
    nonce := make([]byte,12)
    if _,err=rand.Read(nonce);err!=nil { fail("nonce unavailable") }
    suffix := hex.EncodeToString(nonce)
    ready := "MXFP4_TERM_READY_"+suffix
    end := "MXFP4_TERM_EXIT_"+suffix+"="
    collected := ""
    readUntil := func(pattern *regexp.Regexp) []string {
        for {
            if match:=pattern.FindStringSubmatch(collected);match!=nil { return match }
            _, data, err := socket.ReadMessage()
            if err!=nil { fail("terminal interrupted; outcome unknown; no automatic retry") }
            collected+=string(data)
            if len(collected)>1<<20 { fail("terminal output exceeded bound") }
        }
    }
    send("stty -echo; unset HISTFILE; HISTSIZE=0; set +o history; printf '\\n"+ready+"\\n'\n")
    readUntil(regexp.MustCompile("(?:^|[\\r\\n])"+ready+"[\\r\\n]"))
    collected=""
    remote := "/run/mxfp4-terminal-"+suffix
    send("umask 077; : > "+remote+".b64\n")
    payload:=base64.StdEncoding.EncodeToString(source)
    for len(payload)>0 {
        length:=400
        if len(payload)<length { length=len(payload) }
        send("printf %s '"+payload[:length]+"' >> "+remote+".b64\n")
        payload=payload[length:]
    }
    command:="base64 -d "+remote+".b64 > "+remote+".sh && test \"$(sha256sum "+remote+".sh | cut -d' ' -f1)\" = "+os.Args[2]+" && bash "+remote+".sh; mxfp4_rc=$?; printf '\\n"+end+"%s\\n' \"$mxfp4_rc\"; exit\n"
    send(command)
    match:=readUntil(regexp.MustCompile("(?:^|[\\r\\n])"+end+"([0-9]{1,3})[\\r\\n]"))
    code,_:=strconv.Atoi(match[1])
    result:=map[string]any{"job":job,"pod":pod,"owner":"jincy26","script_sha256":os.Args[2],"script_bytes":len(source),"script_exit_code":code,"authenticated_owned_job_verified":true,"token_or_password_output":false,"transcript":strings.TrimSpace(collected)}
    if json.NewEncoder(os.Stdout).Encode(result)!=nil { fail("could not write terminal result") }
    if code!=0 { os.Exit(1) }
}
