//package com.prism.security_core.auth;

package com.prism.security_core.auth;

public class AuthRequest {

    private String username;
    private String password;

    // getters & setters
    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
