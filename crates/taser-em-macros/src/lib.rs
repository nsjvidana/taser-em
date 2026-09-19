use proc_macro::{TokenStream};
use proc_macro2::{TokenStream as TokenStream2, TokenTree as TokenTree2, Ident as Ident2};
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Ident};
use syn::spanned::Spanned;

/// Clones a function with all instances of identifiers named `axis` `axis1` and `axis2` in its
/// body replaced by the attribute values specified in this macro's invocation.
///
/// `axis` will also be placed as a suffix of the function clones.
///
/// # Example
/// The following function:
/// ```
/// #[clone_replaced_axes(axis = z, axis1 = x, axis2 = y)]
/// fn axis_stuff(b: UVec3) -> f32 {
///     b.axis * (b.axis2 - b.axis1)
/// }
/// ```
/// gets cloned as:
///
/// ```
/// fn axis_stuff_z(b: UVec3) -> f32 {
///     b.z * (b.y - b.x)
/// }
/// ```
#[proc_macro_attribute]
pub fn clone_replaced_axes(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);

    let mut fn_name_suffix = String::new();
    let mut axis = None;
    let mut axis1 = None;
    let mut axis2 = None;
    let attr_parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("axis") {
            let axis_ident: Ident = meta.value()?.parse()?;
            fn_name_suffix = axis_ident.to_string();
            axis = Some(axis_ident);
            Ok(())
        } else if meta.path.is_ident("axis1") {
            let axis1_ident: Ident = meta.value()?.parse()?;
            axis1 = Some(axis1_ident);
            Ok(())
        } else if meta.path.is_ident("axis2") {
            let axis2_ident: Ident = meta.value()?.parse()?;
            axis2 = Some(axis2_ident);
            Ok(())
        } else {
            Err(meta.error(format!("Unsupported attribute: {}", meta.path.clone().into_token_stream().to_string())))
        }
    });
    parse_macro_input!(attr with attr_parser);

    if axis.is_none() || axis1.is_none() || axis2.is_none() {
        return syn::Error::new(func.span(), "Expected axis, axis1, and axis2 attributes")
            .to_compile_error().into();
    }

    let og_name = func.sig.ident.to_string();
    let new_name = format!("{og_name}_{fn_name_suffix}");
    let new_ident = Ident::new(&new_name, func.sig.ident.span());
    let mut func_clone = func.clone();
    func_clone.sig.ident = new_ident;

    let a_replaced = replace_idents_in_stream(
        func_clone.block.into_token_stream(),
        "axis",
        axis.unwrap().to_string().as_str(),
    );
    let a1_replaced = replace_idents_in_stream(
        a_replaced,
        "axis1",
        axis1.unwrap().to_string().as_str(),
    );
    let new_func_block = replace_idents_in_stream(
        a1_replaced,
        "axis2",
        axis2.unwrap().to_string().as_str(),
    );
    func_clone.block = syn::parse2(new_func_block).unwrap();
    func_clone.attrs.retain(|a| {
        !a.path().is_ident("clone_replaced_axes")
    });

    let n_self_attrs = func.attrs.iter()
        .filter(|a| a.path().is_ident("clone_replaced_axes"))
        .count();

    if n_self_attrs != 0 {
        TokenStream::from(quote! {
            #func
            #func_clone
        })
    }
    else {
        TokenStream::from(quote! {
            #func_clone
        })
    }
}

fn replace_idents_in_stream(stream: TokenStream2, target: &str, replacement: &str) -> TokenStream2 {
    let mut new_stream = TokenStream2::new();

    for tt in stream {
        match tt {
            TokenTree2::Ident(ref ident) if ident == target => {
                let new_ident = Ident2::new(replacement, ident.span());
                new_stream.extend(quote! { #new_ident });
            }
            TokenTree2::Group(group) => {
                let inner_stream = replace_idents_in_stream(group.stream(), target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner_stream);
                new_group.set_span(group.span());
                new_stream.extend(quote! { #new_group });
            }
            other => new_stream.extend(quote! { #other })
        }
    }

    new_stream
}