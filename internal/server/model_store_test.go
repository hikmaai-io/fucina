package server

import (
	"testing"

	"github.com/hikmaai-io/fucina/internal/model"
)

func TestModelStorePublishedForFutureDetailRoute(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	eng := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(eng, tk)
	if srv.ModelStore() != nil {
		t.Fatal("new server unexpectedly has a model store")
	}
	store := model.NewStore(nil)
	srv.SetModelStore(store)
	if got := srv.ModelStore(); got != store {
		t.Fatalf("ModelStore()=%p want %p", got, store)
	}
}
