package chat

import "testing"

func TestSingleUserTurn(t *testing.T) {
	got := Render([]Message{{Role: "user", Content: "hello"}}, false, "", nil)
	want := "<|turn>user\nhello<turn|>\n" + ModelTurnOpenNoThink
	if got != want {
		t.Errorf("single user turn:\n got: %q\nwant: %q", got, want)
	}
}

func TestSystemAndUser(t *testing.T) {
	got := Render([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
	}, false, "", nil)
	want := "<|turn>system\nbe brief<turn|>\n" +
		"<|turn>user\nhi<turn|>\n" +
		ModelTurnOpenNoThink
	if got != want {
		t.Errorf("system+user:\n got: %q\nwant: %q", got, want)
	}
}

func TestMultiTurnWithAssistant(t *testing.T) {
	got := Render([]Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "q2"},
	}, false, "", nil)
	want := "<|turn>user\nq1<turn|>\n" +
		"<|turn>model\na1<turn|>\n" +
		"<|turn>user\nq2<turn|>\n" +
		ModelTurnOpenNoThink
	if got != want {
		t.Errorf("multi-turn:\n got: %q\nwant: %q", got, want)
	}
}

func TestThinkingOff(t *testing.T) {
	got := Render([]Message{{Role: "user", Content: "hi"}}, false, "", nil)
	if contains(got, "<|think|>") {
		t.Errorf("thinking off must NOT contain <|think|>: %q", got)
	}
	if !endsWith(got, ModelTurnOpenNoThink) {
		t.Errorf("thinking off must end with no-think open: %q", got)
	}
}

func TestThinkingOn(t *testing.T) {
	got := Render([]Message{{Role: "user", Content: "hi"}}, true, "", nil)
	if !contains(got, "<|think|>") {
		t.Errorf("thinking on must contain <|think|>: %q", got)
	}
	if !endsWith(got, ModelTurnOpenThink) {
		t.Errorf("thinking on must end with think open: %q", got)
	}
	// Thinking on forces a system turn even with no system message.
	if !startsWith(got, "<|turn>system\n<|think|>\n") {
		t.Errorf("thinking on must force a system turn with <|think|>: %q", got)
	}
}

func TestSysExtraInjected(t *testing.T) {
	got := Render([]Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hi"},
	}, false, "TOOLDECL", nil)
	want := "<|turn>system\nsysTOOLDECL<turn|>\n" +
		"<|turn>user\nhi<turn|>\n" +
		ModelTurnOpenNoThink
	if got != want {
		t.Errorf("sysExtra injection:\n got: %q\nwant: %q", got, want)
	}
	// SysExtra alone forces the system turn even without a system message.
	got2 := Render([]Message{{Role: "user", Content: "hi"}}, false, "TOOLDECL", nil)
	if !startsWith(got2, "<|turn>system\nTOOLDECL<turn|>\n") {
		t.Errorf("sysExtra must force a system turn: %q", got2)
	}
}

func TestTrailingModelOpenPresent(t *testing.T) {
	got := Render([]Message{{Role: "user", Content: "hi"}}, false, "", nil)
	if !endsWith(got, ModelTurnOpenNoThink) {
		t.Errorf("expected trailing model open: %q", got)
	}
}

func TestLastEmptyAssistantOpensTurn(t *testing.T) {
	got := Render([]Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: ""},
	}, false, "", nil)
	want := "<|turn>user\nhi<turn|>\n" + ModelTurnOpenNoThink
	if got != want {
		t.Errorf("last empty assistant must open (not close) the turn:\n got: %q\nwant: %q", got, want)
	}
}

func TestTurnExtraHook(t *testing.T) {
	// turnExtra is appended inside a non-trailing assistant turn before <turn|>.
	got := Render([]Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "q2"},
	}, false, "", func(i int) string {
		if i == 1 {
			return "CALLS"
		}
		return ""
	})
	if !contains(got, "<|turn>model\naCALLS<turn|>\n") {
		t.Errorf("turnExtra must append inside model turn: %q", got)
	}
}

// tiny helpers to avoid importing strings in assertions.
func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func startsWith(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }
func endsWith(s, p string) bool   { return len(s) >= len(p) && s[len(s)-len(p):] == p }
