#include <bits/stdc++.h>

using namespace std;
using vi = vector<int>;

void solve() {
    int cnt = 0;
    string s;
    cin >> s;
    for(int i = 0; i < s.length(); i ++){
        if(s[i] == 'N'){
            cnt++;
        }
        if(cnt >= 2){
            break;
        }
    }
    cout << (cnt == 1 ? "NO" : "YES") << "\n";

}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int t;
    cin >> t;
    while (t--) solve();
}