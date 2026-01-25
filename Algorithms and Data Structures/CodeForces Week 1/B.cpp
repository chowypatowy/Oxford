#include <bits/stdc++.h>

using namespace std;
using vi = vector<int>;
#define pb push_back

vi R;
vector<vi> Rinv;

void make_set(int x){
    R[x] = x;
    Rinv[x] = {x};
}

int find_set(int x){
    return R[x];
}

void combine(int x, int y){
    if(R[x] == R[y]){
        return;
    }
    int rx = R[x];
    int ry = R[y];
    for(int i : Rinv[rx]){
        R[i] = ry;
        Rinv[ry].pb(i);
    }
    Rinv[rx] = {};
}

void print_dsu(int n) {
    cout << "R: [";
    for(int i = 0; i < n; i++) {
        cout << R[i] << (i == n - 1 ? "" : ", ");
    }
    cout << "]" << endl;
}

void solve() {
    int n; cin >> n;
    R.assign(n, -1);
    Rinv.assign(n, {});
    for(int i = 0; i < n; i ++){
        make_set(i);
    }
    for(int i = 0; i < n - 1; i ++){
        int a, b;
        cin >> a; cin >> b;
        a -= 1; b -= 1;
        combine(a, b);
        print_dsu(n);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int t = 1;
    while (t--) solve();
}
