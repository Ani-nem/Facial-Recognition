"use client";
import React, {useState, useContext, ChangeEvent} from "react";
import {Button} from "@/components/ui/button";
import {Input} from "@/components/ui/input";
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import {useRouter} from "next/navigation";
import {get_user_info, login, register, UserContext} from "@/utils/auth";
import {Label} from "@/components/ui/label";
import {Tabs, TabsContent, TabsList, TabsTrigger} from "@/components/ui/tabs";

const LoginForm = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [verifyPassword, setVerifyPassword] = useState("")
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const router = useRouter();
    const context = useContext(UserContext);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!email || !password) {
            setError("Missing login or password")
        } else {
            setLoading(true);
            setError("");
            const success = await login(email, password);
            setLoading(false);

            if (success) {
                const userInfo = await get_user_info();
                context?.setUser(userInfo);
                router.push("/dashboard");
            } else {
                setError("Incorrect Email or Password");
            }
        }

    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        if (verifyPassword !== password) {
            setError("Passwords do not match");
        } else {
            setLoading(true);
            setError("");
            const success = await register(email, password);
            setLoading(false);

            if (success) {
                await handleLogin(e); // Call handleLogin after successful registration
            } else {
                setError("Registration failed");
            }
        }
    };

    const onVerifyPasswordChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.value !== password) {
            setError("Passwords do not match");
        } else {
            setError("");
        }
        setVerifyPassword(event.target.value)
    };
    return (
        <Tabs defaultValue={"login"} className={"w-full max-w-sm"}>
            <TabsList className={"grid w-full grid-cols-2"}>
                <TabsTrigger value={"login"}>Login</TabsTrigger>
                <TabsTrigger value={"register"}>Register</TabsTrigger>
            </TabsList>
            <TabsContent value={"login"}>
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle className={"text-2xl"}>Login</CardTitle>
                        <CardDescription>Enter your email and password to login to your account.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleLogin} noValidate={true}>
                            <div className="space-y-4">
                                <div className={"grid gap-2"}>
                                    <Label>Email</Label>
                                    <Input
                                        type="email"
                                        placeholder="Email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                    />
                                </div>

                                <div className={"grid gap-2"}>
                                    <Label>Password</Label>
                                    <Input
                                        type="password"
                                        placeholder="Password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                </div>

                                {error && (
                                    <div className="text-red-700 text-sm">{error}</div>
                                )}
                                <Button type="submit" className="w-full" disabled={loading}>
                                    {loading ? "Logging in..." : "Login"}
                                </Button>
                            </div>
                        </form>
                    </CardContent>
                </Card>
            </TabsContent>
            <TabsContent value={"register"}>
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle className={"text-2xl"}>Register</CardTitle>
                        <CardDescription>Don&#39;t have an account? Enter an email and password to
                            register.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleRegister} noValidate={true}>
                            <div className="space-y-4">
                                <div className={"grid gap-2"}>
                                    <Label>Email</Label>
                                    <Input
                                        type="email"
                                        placeholder="Email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                    />
                                </div>

                                <div className={"grid gap-2"}>
                                    <Label>Password</Label>
                                    <Input
                                        id={"password"}
                                        type="password"
                                        placeholder="Password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                </div>

                                <div className={"grid gap-2"}>
                                    <Label>Verify Password</Label>
                                    <Input
                                        id={"verify password"}
                                        type="password"
                                        placeholder="Re-enter Password"
                                        onChange={(e) => onVerifyPasswordChange(e)}
                                        onPaste={(e) => e.preventDefault()}
                                        required
                                    />
                                </div>


                                {error && (
                                    <div className="text-red-700 text-sm">{error}</div>
                                )}
                                <Button type="submit" className="w-full" disabled={loading}>
                                    {loading ? "Logging in..." : "Register"}
                                </Button>
                            </div>
                        </form>
                    </CardContent>
                </Card>
            </TabsContent>
        </Tabs>

    );
}

export default LoginForm;